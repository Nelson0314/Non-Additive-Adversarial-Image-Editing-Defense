使用者於 2026-08-17 提供的九張原始素材，放在 repo 根目錄。
scripts/ 沒有轉檔腳本——轉檔規則記在這裡：

  置中裁成正方（min(w,h)）→ LANCZOS 縮到 512x512 → 存成 PNG

置中裁切是必要的：load_image_tensor 的 size 參數走 F.interpolate 直接拉成
size x size，非正方的素材會被壓扁，而那個變形不會有任何症狀。

對照：
  Barack-Obama-2012.webp             -> obama/obama_00.png
  James_l_thumbnail_908893.jpg       -> lebron/lebron_00.png
  images (1).jpg                     -> ronaldo/ronaldo_00.png
  images (2).jpg                     -> musk/musk_00.png
  images.jpg                         -> trump/trump_00.png
  images (3).jpg                     -> parrot/parrot_00.png
  istockphoto-1443562748-612x612.jpg -> cat/cat_00.png
  raccoon-grass_3x2.avif             -> raccoon/raccoon_00.png
  shiba-inu.jpg                      -> shiba/shiba_00.png
