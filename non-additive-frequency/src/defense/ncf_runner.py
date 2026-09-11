"""IP2P adapter: strict experiment provenance, unresized original RGB, L_IG only."""
import torch

from .carrier_mask import (MAX_AREA, MIN_AREA, carrier_mask, erode_mask,
                           exclude_boxes, feather_inward, guided_refine)


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

