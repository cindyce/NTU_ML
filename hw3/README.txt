本次作業嘗試使用的model有densenet201、resnext50_32x4d

執行r10945001_densenet201.py :
曾經使用AutoAugmentation做transforms(後來沒用)
引用AutoAugment: https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py


另外我也有將model產生的多個不同的預測結果(參數調整)做voting的ensemble，此需產生多個submission.csv(檔名開頭須為submission)執行ensemble.py


r10945001_resnext50_32x4d.py :
(另外的版本，因為時間有限還沒跑完，可以不用執行)


