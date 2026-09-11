"""IP2P adapter: strict experiment provenance, unresized original RGB, L_IG only."""
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import torch

from .carrier_mask import (MAX_AREA, MIN_AREA, carrier_mask, erode_mask,
                           exclude_boxes, feather_inward, guided_refine)
from .ncf_library import NCFLibrary, sha256
from .ncf_search import NCFConfig, NCF_CONDITIONS, search, write_csv


NCF_REGIONS = ('frame', 'clothes')

# The values the clothing carrier was validated at, shared by every dispatch
# script in this repo and by the snapshot before the cleanup (9 and 11 scripts
# respectively, all `--carrier-refine 4 --carrier-erode 3 --carrier-feather 8`).
# They are written into the CSV rather than left as call-site literals.
CARRIER_REFINE, CARRIER_ERODE, CARRIER_FEATHER = 4, 3, 8


def ncf_support(x, region):
    """(1,1,H,W) in [0,1]. 1 = may be recoloured, 0 = copied from the original.

    Only two regions are supported, and neither is the face/hair complement the
    first run used. That complement was a third thing — clothing plus background
    plus neck plus hands — and it inherited every ATR error: a couch cushion
    read as hair, an unprotected nose, a jagged beard boundary.

    `frame`   the whole image; a colour filter with no mask at all.
    `clothes` the ATR clothing classes put through the validated carrier path:
              guided filter onto the real garment edge, MTCNN face boxes cut
              out, eroded inward, feathered inward. Every step only ever
              removes weight, so nothing outside the garment can be touched.
    """
    if region == 'frame':
        return torch.ones_like(x[:, :1])
    if region != 'clothes':
        raise ValueError(f'region must be one of {NCF_REGIONS}, got {region!r}')
    from src.metrics.identity import face_boxes

    c = carrier_mask(x, 'clothes', min_area=MIN_AREA, max_area=MAX_AREA)
    c = (guided_refine(c, x, CARRIER_REFINE, 1e-3) > 0.5).to(c.dtype)
    # ATR marks a cheek or forehead as Upper-clothes often enough that the face
    # box is a separate, independent signal and not a belt-and-braces extra.
    c = exclude_boxes(c, face_boxes(x), margin=8)
    c = erode_mask(c, CARRIER_ERODE)
    return feather_inward(c, CARRIER_FEATHER)


def validate_ncf_args(args):
    if any(c not in NCF_CONDITIONS for c in args.conditions):
        raise ValueError('run NCF conditions separately to retain original image resolution')
    if args.loss != 'image_guidance' or args.ig_zt != 'diffuse_src' or args.ig_weight != 'uniform':
        raise ValueError('NCF requires unweighted L_IG with diffuse_src')
    # These options alter the registered objective/output and cannot be silently ignored.
    for key in ('prompt_eot', 'blend_latent_norm', 'manifold_weight', 'manifold_only',
                'attn_weight', 'stage2_steps', 'color_tv', 'color_tv_luma', 'patch_tv',
                'flow_tau', 'consistency_weight', 'resume_weights', 'subject_mask',
                'deliver_jpeg', 'deliver_linf'):
        if getattr(args, key, None):
            raise ValueError(f'NCF L_IG protocol does not support --{key.replace("_", "-")}')
    spec = json.loads(args.ncf_config.read_text(encoding='utf-8'))
    if spec.get('region') not in NCF_REGIONS:
        raise ValueError(f"NCF config must set region to one of {NCF_REGIONS}; "
                         "the face/hair complement is not an accepted region")
    manifest_path = Path(spec['manifest'])
    if sha256(manifest_path) != spec['manifest_sha256']:
        raise ValueError('NCF manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not args.images or not set(args.images) <= {r['id'] for r in manifest['images']}:
        raise ValueError('NCF --images must explicitly select the five registered images')
    if args.ncf_resume is not None and (len(args.images) != 1 or len(args.conditions) != 1):
        raise ValueError('NCF resume requires exactly one image and condition')
    if args.attack_category is None or str(args.attack_prompts).replace('\\', '/') != spec['attack_prompts']:
        raise ValueError('NCF requires the registered attack-prompts file and category')
    if sha256(args.attack_prompts) != spec['attack_prompts_sha256']:
        raise ValueError('NCF attack instruction hash mismatch')
    for r in manifest['images']:
        if sha256(r['path']) != r['sha256']:
            raise ValueError(f'NCF input hash mismatch: {r["id"]}')
        if any(s % 8 for s in Image.open(r['path']).size):
            raise ValueError('NCF original dimensions must be divisible by 8; no silent resize')


def defend_ncf(condition, x, args, loss):
    spec = json.loads(args.ncf_config.read_text(encoding='utf-8'))
    cfg = NCFConfig(**spec['search'])
    manifest = json.loads(Path(spec['manifest']).read_text(encoding='utf-8'))
    entry = next(r for r in manifest['images'] if r['id'] == args._cur_image)
    region = spec['region']
    support = ncf_support(x, region)
    extras = {'_ncf_support': support, 'ncf_split': entry['split'],
              'ncf_input_sha256': entry['sha256'], 'ncf_config_sha256': sha256(args.ncf_config),
              'ncf_region': region, 'ncf_support_area': float(support.float().mean()),
              'ncf_carrier_refine': CARRIER_REFINE if region == 'clothes' else '',
              'ncf_carrier_erode': CARRIER_ERODE if region == 'clothes' else '',
              'ncf_carrier_feather': CARRIER_FEATHER if region == 'clothes' else '',
              'ncf_ade20k_classes': '|'.join(spec.get('ade20k_classes') or []),
              'ncf_loss_normalised': int(bool(spec.get('normalise_loss'))),
              'ncf_library_sha256': spec['library_sha256'], **{'ncf_'+k: v for k, v in asdict(cfg).items()}}
    folder = args.out / f'{args._cur_image}__{condition}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(support.cpu(), folder/'support.pt')
    if condition == 'ncf_clean':
        return x.detach().clone(), 0., False, False, extras
    library = NCFLibrary(spec['library'], expected_sha256=spec['library_sha256'],
                         required_classes=spec['class_weights'], cov_floor=cfg.cov_floor,
                         ade20k_classes=spec.get('ade20k_classes'))
    write_csv(folder/'library_audit.csv', library.audit)
    cp = None if args.ncf_resume is None else torch.load(args.ncf_resume, map_location=x.device, weights_only=False)
    result, state = search(x, support, library, loss, condition=condition, config=cfg,
                           weights=spec['class_weights'], out=folder, checkpoint=cp)
    extras.update({'ncf_best_fixed_ig': state['best_loss'], 'ncf_steps_completed': state['global_step'],
                   'ncf_records_in_pool': library.metadata['records_in_pool'],
                   'ncf_source': library.metadata['source']})
    return result, cfg.epsilon, False, True, extras
