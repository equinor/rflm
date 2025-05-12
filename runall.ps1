python .\rflm_script.py -i ..\sharepoint\data_Lassen2005.txt -o Lassen2005 -v 1 --nlim 1E4 1E8
python .\rflm_script.py -i ..\sharepoint\data_Stojkovic2018.mat -o Stojkovic2018 -v 1 --nlim 1E4 1E8
python .\rflm_script.py -i ..\sharepoint\data_Mikulski2022.mat -o Mikulski2022 -v 1 --nlim 1E4 1E8
python .\rflm_script.py -i ..\sharepoint\DNV_as_welded_filtered.txt -o DNV_as_welded_filtered -v 1 --nlim 1E4 1E8
#Run quantile model
python .\run_quantile_model.py -i .\data\data_syn2.xlsx -p .\parameters\syn2.xlsx --quantiles 0.025 0.5 0.99 --slim 40 450 --nlim 1e4 1e8