本資料集的原始素材與轉檔規則。scripts/ 沒有轉檔腳本——規則記在這裡：

  置中裁成正方（min(w,h)）→ LANCZOS 縮到 512x512 → 存成 PNG

置中裁切是必要的：load_image_tensor 的 size 參數走 F.interpolate 直接拉成
size x size，非正方的素材會被壓扁，而那個變形不會有任何症狀。

---- 2026-08-18：三張無名氏肖像（取代五張名人照）----

全部取自 Wikimedia Commons，授權 CC0，均為真實照片（不得用生成影像）。

  person_a/person_a_00.png
    File:Portrait of a man (Unsplash).jpg   4096x2732
    攝影 William Stitt，2016-09-18
    https://commons.wikimedia.org/wiki/File:Portrait_of_a_man_(Unsplash).jpg
    原始出處 https://unsplash.com/photos/B0m8ZIR-CF0

  person_b/person_b_00.png
    File:Face portrait (Unsplash).jpg       5184x3456
    攝影 William Stitt，2016-10-19
    https://commons.wikimedia.org/wiki/File:Face_portrait_(Unsplash).jpg
    原始出處 https://unsplash.com/photos/PgSu429ID5o

  person_c/person_c_00.png
    File:Brunette woman portrait (Unsplash).jpg  5184x3456
    攝影 Christopher Campbell，2015-10-21
    https://commons.wikimedia.org/wiki/File:Brunette_woman_portrait_(Unsplash).jpg
    原始出處 https://unsplash.com/photos/3hoAon9Mc88

CC0 不要求標示出處，此處記錄是為了讓素材可追溯、可重新取得。

---- 2026-08-17：使用者提供的素材 ----

  images (3).jpg                     -> parrot/parrot_00.png
  istockphoto-1443562748-612x612.jpg -> cat/cat_00.png
  raccoon-grass_3x2.avif             -> raccoon/raccoon_00.png
  shiba-inu.jpg                      -> shiba/shiba_00.png

同批的五張名人照（Barack-Obama-2012.webp、James_l_thumbnail_908893.jpg、
images (1).jpg、images (2).jpg、images.jpg）已於 2026-08-18 隨 obama／lebron／
ronaldo／musk／trump 五個目錄一併移除，取回：
  git checkout 2d23704b1 -- data/set0817
