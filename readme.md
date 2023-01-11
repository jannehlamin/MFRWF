# 
<h2>1.  Multi-Level Feature Re-weighted fusion for the Semantic Segementation of crops and weeds </h2>
<p>
 A proposed network for the pixel-wise semantic segmentation of crops and weed with minimal memory overhead, it is experimented on three commonly 
 used datasets
</p>

<h3>2. Network Training and Testing </h3>
<p>
It was trained and tested using:
 <ul>
   <li> NVIDIA Tesla P40 GPUs</li>
   <li> PyTorch 1.11.0 </li>
 </ul>
</p>



<h3>3. Dataset and the Experimental results </h3>
<p>
 The datasets for testing and the model files must in the project root directory, which can be access via the following links to google drive
 <ul>
  <li> <a href="https://drive.google.com/drive/folders/1ojbzjbHcu1dR3GcGrzgohdKLh1-zjBzr?usp=share_link">Test datasets and model files</a> </li>
 </ul>
</p>

<h3> Running commands for the Test </h3>
<p>
The experiment can be tested using the following commands, where the dataset parameter can be changed for the different datasets
 bweeds : Bonirob , cweeds : CWFID, rweeds : Rice seedlings 
</p>

<p>
 <ul>
  <li> <b>Baseline</b> (bweeds, cweeds, rweeds) </li>
   <li> cmd: <i>python main_ours.py --dataset='bweeds' --backbone='baseline' </i> </li>
 </ul>
 
 <ul>
   <li> <b>MFF</b> (bweeds, cweeds, rweeds) </li>
    <li> cmd:  <i>python main_ours_nostream.py --dataset='bweeds' --backbone='ours_l34rw_partial_weight' </i></li>
 </ul>
 <ul>
   <li> <b>MFRWF</b> (bweeds, cweeds, rweeds) </li>
   <li> cmd:  <i>python main_ours_nostream.py --dataset='bweeds' --backbone='ours_l34rw_partial_decoder' </i></li>
 </ul>
 <ul>
 <li> <b>MFRWF + CWF</b> (bweeds, cweeds, rweeds) </li>
 <li> cmd:  <i>python main_ours_nostream.py --dataset='bweeds' --backbone='ours_l34rw_fully' </i></li>
 </ul>
</p>
<hr />
<hr />
<h3>Visulaization of the Results</h3>
 <ul>
   <li> Bonirob: <br />
    <img src="https://github.com/jannehlamin/MFRWF/blob/master/result_img/bonirob.png" alt="bonirob datset" width="1024" height="512" />
   </li>
   <li> CWFID:<br />
        <img src="https://github.com/jannehlamin/MFRWF/blob/master/result_img/cwfid.png" alt="CWFID datset" width="1024" height="512" />
   </li>
    <li> Rice Seedlings:<br />
        <img src="https://github.com/jannehlamin/MFRWF/blob/master/result_img/rs.jpg" alt="Rice seedlings datset" width="1024" height="512" />
   </li>
 </ul>
