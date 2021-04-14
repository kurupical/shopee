# 2021/4/1
* 今日から頑張る
* [TODO] foldの切り方。ランダムなのか?未知ラベルでvalidationする方がいいのか?
    * とりあえず前者でやってlossとか計算する。。
* BERTはそんなに学習率高くなくていい気がする

# 2021/4/3
https://www.mindmeister.com/1844996822?e=turtle
↑でタスク管理

* CNN単体だとうまくいくが, bertだとうまくいかない。。
 -> BERTの取り入れ方をいろいろいじる
  　①cnnとbertで別口のfcを用意→ダメ
  　②automodelの[1]を抽出(なんだこれ)してたのを、[0]、つまり普通の層を出力し平均とったらまだましだった
  

* csvデータはインドネシア語
    * bert-base-multilingual-uncasedを使うとどうだろう?

# 2021/4/4
* exp002は未知ラベル少し加えてミックス
* exp003はvalを未知ラベルだけにして同じ実験してみる
* exp003より, base_lrも小さい方がいいということがわかった！

# 2021/4/5
* exp006: train/valを分ける, 10foldでやる

# 2021/4/6
* stable cvを考える
  * なんでCV-LBで乖離があるのか？　→　データ分布を見てみよ
    * そもそもデータ分布
    * グループ1件のときや2件のときの件数と精度
  * kfold(3)とかでやってみるとどうだろう。。(->exp007)
  * kfold(3)のアンサンブルをとるようなのやってみよう

# 2021/4/7
* kfold(2) exp007 -> bert CV:0.787 - LB:0.649
* kfold(2) with euclidean distance
* image_phashの名寄せは万能じゃない 0.7887 -> 0.7875
* text名寄せはいいかも　0.7887 -> 0.7907 (+0.02)
![img.png](img.png)


# 2021/4/8
* EDA2: 全画像のペア見る　→　Augmentation考察
  * randomcrop
    ![img_1.png](img_1.png)
  * centorcrop
  * 人だけとってくる
    ![img_2.png](img_2.png)
  * メモ：大文字と小文字は区別要らなさそう
  * 角度回転して確認してみたい。。
    ![img_3.png](img_3.png)
  
* exp008: https://www.kaggle.com/c/shopee-product-matching/discussion/228794　これでfold切ってみる。。

* exp010: andomcrop

# 2020/4/11
* あってるやつと間違ってるやつの距離差
![img_4.png](img_4.png)
  
* exp013: BERT+image(512*512)
  CV: 0.837(fold0) -> LB: 0.727 (th=15)
* exp014: BERT-CNNでdropout平等 + いろいろ実験

# 2021/4/12
* cpptakeさんに教えてもらったAdacosをベースに論文調査
* exp016: bert-base-indonesian-522M -> そんな変わらん

* exp017: distanceを正規分布過程で疑似的に65000件生成し、70000件での画像検索を想定する
![img_5.png](img_5.png)
  
* exp018: exp013 + OCR -> そんなに

* exp019: いろんなモデル試す + アンサンブルできるようにする
* exp020: アンサンブル実験用コード(fold同じ)
* exp021: CNN -> BERT

# 2021/4/14
* batch_sizeがかなり学習結果に影響してそう。batch_size=8のものは軒並み精度が出ていない気がする。
  -> LRを増やさないと学習進まない? bs=16とbs=8だったら後者がlr2倍か?小さいモデルで実験してみよう
  -> exp022
  
* optimize_ensemble_v1_20210414
  * single best: CV 0.856
  * ensemble major: threshold 18.5 -> CV 0.8764
  
* exp022
  bs [8, 12, 16] -> CV: 0.8067 - 0.8216 - 0.8271
  bs [8, 12, 16] (lr調整:減らす) -> だめ
  bs [8, 12, 16] (lr調整:増やす) -> CV: 0.8067 -
  
