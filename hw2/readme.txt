本次作業使用的model是GRU，須將輸入data轉換為3維的資料，Dropout設置為0.5。

多次執行r10945001_hw2.py，並微幅調整參數(concat_nframes, seed, layers...)，得到多個prediction結果再執行ensemble.py。