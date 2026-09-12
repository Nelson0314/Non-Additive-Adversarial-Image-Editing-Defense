# 過渡帶：讓背景的約束只管背景

派工 `scripts/immunise_field.py`，設定 `configs/immunise_field_margin.json`，
由 `scripts/batch_queue.sh` 排程。自變數是**過渡帶的寬度**（`rigid_margin`），
其餘與 `runs/field_affine/` 相同。

## 前一批量到的瓶頸

`runs/field_affine/` 的 `cap_reached` 說得很清楚。`face_rigid_16` 允許臉框內
位移的 CVaR99 走到 16 px，五張實際只走到 **1.42–2.01**；而背景那道
`flow_rigid` 貼死在 **1.98–1.998 / 2**。綁住解的是背景剛性，不是臉框預算。

原因在支撐的畫法：`flow_rigid` 的支撐直接取臉框的補集，於是**臉框外緣的
過渡帶也被算成背景**。位移場是平滑的，臉要整體移動就必然拉動它周圍一圈，
而那一圈落在臉框外——背景約束因此把臉的移動一起擋掉。兩道約束互相打架。

這在第一批看不出來：`flow_16` 沒有臉框內的仿射約束，臉可以用局部形變吃掉
預算（`flow_face` 走到 11.6–15.9），而局部形變在臉框內衰減得快，溢出到背景的
部分小。加上仿射約束之後，臉只剩整體移動，而整體移動必然溢出。

## 修法

`geometry_field.dilate` 把臉框向外膨脹 `rigid_margin` 像素再取補集。
過渡帶因此兩邊都不管：**它既不是要保直線的背景，也不是要保形狀的臉。**

膨脹用可分離的最大值濾波（方形結構元），`radius = 0` 是恆等，
`tests/test_field_carriers.py` 釘住這三件事：只會變大不會變小、半徑為零是恆等、
半徑過大時背景的支撐會變空（那時約束沒有定義，腳本直接拒絕啟動而不是靜默跑過）。

## 五個變體

| 變體 | 臉框 | 臉內仿射 | 背景 | 過渡帶 |
|---|---|---|---|---|
| `margin_00` | 16 px | 1.0 px | 2.0 px | 0 px |
| `margin_32` | 16 px | 1.0 px | 2.0 px | 32 px |
| `margin_64` | 16 px | 1.0 px | 2.0 px | 64 px |
| `margin_64_loose` | 32 px | 2.0 px | 2.0 px | 64 px |
| `margin_64_norigid` | 32 px | — | 2.0 px | 64 px |

`margin_00` 與 `runs/field_affine/` 的 `face_rigid_16` 逐字相同，是這一批自己的
對照：過渡帶寬度為零就是原來的作法，兩批之間若有環境差異它會顯示出來。

`margin_64_norigid` 拿掉臉框內的仿射約束，只留過渡帶。它是
`runs/field_affine/` 那個假設的直接對照——若它比 `margin_64` 強很多而圖一樣
自然，代表臉框內的仿射約束買到的自然度不值那個防禦。

## 怎麼讀

先看 `cap_reached` 裡 `flow_face` 這一格：如果它隨過渡帶加寬而逼近上限，
瓶頸的診斷就對了；如果仍然停在 1–2 px，瓶頸在別的地方，要回頭找。

`cap_names` 的順序是 `flow_face | flow_rigid | flow_face_rigid | flow_fold`
（`margin_64_norigid` 少第三項）。自然度仍然由人眼判定。
