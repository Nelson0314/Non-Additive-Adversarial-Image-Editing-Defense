#!/usr/bin/env bash
# 佇列驗收：`queue_validate.sh <工作>`；結束碼 0 才記為完成。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
exec "$PY" -m immunization_color.cli.validate_queue_job "$1" --project-root "$COLOR_ROOT" --fid-arms ${FID_ARMS:-}
