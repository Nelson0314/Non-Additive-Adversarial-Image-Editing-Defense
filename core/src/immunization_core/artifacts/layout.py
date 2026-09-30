"""由呼叫端明確提供資料、產物與結果根目錄；不探索兄弟專案。"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """解析呼叫端提供的三個根目錄，不檢查或建立檔案。

    data_root 是包含 prompts.yaml 與 masks/ 的資料集根；相對路徑在建構時
    依呼叫端 CWD 固定為絕對路徑。不存在的目錄仍可描述，驗收由使用端負責。
    """

    data_root: Path
    artifacts_root: Path
    results_root: Path

    def __post_init__(self) -> None:
        for field in ("data_root", "artifacts_root", "results_root"):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())

    @property
    def prompts(self) -> Path:
        """資料集的指令檔。"""
        return self.data_root / "prompts.yaml"

    @property
    def masks(self) -> Path:
        """資料集的重繪遮罩目錄；白色表示重繪。"""
        return self.data_root / "masks"


def defended_image(directory: Path, name: str) -> Path:
    """解析既有 __def.png 名稱；缺少或多個匹配立即拒絕。"""
    files = sorted(set(directory.glob(f"{name}__*__def.png")) |
                   set(directory.glob(f"{name}__def.png")))
    if len(files) != 1:
        raise ValueError(f"{directory} 的 {name} 必須對應一張防禦圖，找到 {len(files)} 張")
    return files[0]
