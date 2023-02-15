echo "  This is the results of our proposed method"
#echo "================================ A. CWFID Dataset ==================================================="
#echo "  PFRWF – SD (our) "
#python main_ours_nostream.py --backbone='ours_l34rw_partial_weight' --dataset='cweeds'
#echo "  FFRWF – SD (our) "
#python main_ours_nostream.py --backbone='ours_l34rw_partial_cwffd' --dataset='cweeds'
#echo "  PFRWF – OD (our) "
#python main_ours_nostream.py --backbone='ours_l34rw_partial_decoder' --dataset='cweeds'
#echo "  FFRWF – OD (our) "
#python main_ours_nostream.py --backbone='ours_l34rw_fully' --dataset='cweeds'

echo "================================ B. BoniRob Dataset ==================================================="
echo "  PFRWF – SD (our) "
python main_ours_nostream.py --backbone='ours_l34rw_partial_weight' --dataset='bweeds'
echo "  FFRWF – SD (our) "
python main_ours_nostream.py --backbone='ours_l34rw_partial_decoder' --dataset='bweeds'
echo "  PFRWF – OD (our) "
python main_ours_nostream.py --backbone='ours_l34rw_partial_cwffd' --dataset='bweeds'
echo "  FFRWF – OD (our) "
python main_ours_nostream.py --backbone='ours_l34rw_fully' --dataset='bweeds'

