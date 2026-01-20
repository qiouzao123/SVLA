# SVA Dataset Description
We construct the SVA dataset to benchmark anomaly detection for User-Generated Content (UGC). Traditional surveillance datasets (e.g., UCF-Crime) depend on static continuity. Distinct from them, SVA captures the unconstrained characteristics of online videos. We collected raw clips from platforms such as TikTok. We notice that this introduces specific challenges, including frequent editing cuts and dynamic backgrounds.Examples of harmful content images are presented as follows：


![violent_video50433_frame_7](https://github.com/user-attachments/assets/06670a8f-d746-404e-8504-0681b2a4ffe3) ![violent_video50179_frame_6](https://github.com/user-attachments/assets/dbcdcd77-ad2f-4713-98c9-cc508959c828)  ![violent_video50280_frame_2](https://github.com/user-attachments/assets/4f9441a5-044d-4fa2-ac9d-b6d54e896605)
![blood_video10443_frame_2](https://github.com/user-attachments/assets/9622df4a-b089-4bed-97d7-9f0efe61fe1f)  ![blood_video10671_frame_1](https://github.com/user-attachments/assets/702e0dea-5784-4bbe-90ce-e9d4935f1715) ![blood_video10869_frame_4](https://github.com/user-attachments/assets/1dd0dcef-c3ff-45e4-a668-380335e22d1c)
![smoke_video40029_frame_1](https://github.com/user-attachments/assets/0cff2360-9b98-453b-ac64-21561f74549d) ![smoke_video40073_frame_1](https://github.com/user-attachments/assets/88ef5a4f-13d3-4c90-8fc6-22a998c5787e) ![smoke_video40451_frame_8](https://github.com/user-attachments/assets/8c27cdfc-4273-42cc-9458-054ca2145df7)


<table>
  <thead>
    <tr>
      <th width="15%">Category</th>
      <th width="52%">Description</th>
      <th width="33%">Representative Samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Smoke</strong></td> 
      <td>Captures instances of streamers smoking or displaying tobacco products during broadcasts, which violates platform health guidelines.</td>
      <td>
        <img src="https://github.com/user-attachments/assets/0cff2360-9b98-453b-ac64-21561f74549d" width="150"> <img src="https://github.com/user-attachments/assets/88ef5a4f-13d3-4c90-8fc6-22a998c5787e" width="150">
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Blood</strong></td>
      <td>Contains visual depictions of physical injuries, bleeding, or medical gore scenes.</td>
      <td>
        <img src="https://github.com/user-attachments/assets/9622df4a-b089-4bed-97d7-9f0efe61fe1f" width="150"> <img src="https://github.com/user-attachments/assets/1dd0dcef-c3ff-45e4-a668-380335e22d1c" width="150">
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Violent</strong></td>
      <td>Depicts physical altercations, street fights, riots, or assaults between individuals or groups.</td>
      <td>
        <img src="https://github.com/user-attachments/assets/06670a8f-d746-404e-8504-0681b2a4ffe3" width="150"> <img src="https://github.com/user-attachments/assets/4f9441a5-044d-4fa2-ac9d-b6d54e896605" width="150">
      </td>
    </tr>
    <tr>
      <td align="center"><strong>🤬 Abusive</strong></td>
      <td>Involves aggressive behaviors, verbal harassment, or inappropriate gestures targeted at others.</td>
      <td>
        <img src="你的长链接7" width="190"> <img src="你的长链接8" width="190">
      </td>
    </tr>
    <tr>
      <td align="center"><strong>🔞 Sexy</strong></td>
      <td>Contains sexually suggestive content, inappropriate exposure, or explicit acts violating platform rules.</td>
      <td>
        <img src="你的长链接9" width="190"> <img src="你的长链接10" width="190">
      </td>
    </tr>
    <tr>
      <td align="center"><strong>💸 Money</strong></td>
      <td>Content related to scams, gambling, or displaying large amounts of cash in a suspicious context.</td>
      <td>
        <img src="你的长链接11" width="190"> <img src="你的长链接12" width="190">
      </td>
    </tr>
    <tr>
      <td align="center"><strong>⚖️ Policy</strong></td>
      <td>Includes politically sensitive content, unauthorized political commentary, or symbols that violate platform regulatory policies.</td>
      <td>
        <img src="你的长链接13" width="190"> <img src="你的长链接14" width="190">
      </td>
    </tr>
  </tbody>
</table>



# SVLA
<img width="1969" height="1084" alt="image" src="https://github.com/user-attachments/assets/d169227c-64b3-44fe-91f5-265217db4b4d" />

# Training
### Setup
We provide pre-extracted CLIP features for the UCF-Crime and XD-Violence datasets, which are released as follows:

| Benchmark | HuggingFace | Baidu |
| :--- | :--- | :--- |
| **SVA** | [HuggingFace](https://huggingface.co/datasets/qiouzao/SVA) | [code:6m3p](https://pan.baidu.com/s/1A_uTeMtLtLHLZR53NgKgxQ?pwd=6m3p) |
| **UCF-Crime** | [HuggingFace](https://huggingface.co/datasets/qiouzao/UCF-Crime) | [code:ppc1](https://pan.baidu.com/s/1dccN0aRQQgwsF_Epo1n1hw?pwd=ppc1) |
| **XD-Violence** | [HuggingFace](https://huggingface.co/datasets/qiouzao/XD-Violence) | [code:5s1r](https://pan.baidu.com/s/1iYukvbpxa9YGD2leg3GflQ?pwd=5s1r) |

To run the code locally, you need to modify the following files:
* Update the file paths in `list/Sva_CLIP_rgb.csv` and `list/Sva_CLIP_rgbtest.csv` to point to the datasets you downloaded earlier.
* Feel free to tweak the hyperparameters in `Sva_option.py` to suit your needs.

### Train and Test
Upon completion of the configuration process, execute the following command:

Traing and infer for SVA dataset
```bash
python Sva_train.py
python Sva_test.py
```
Traing and infer for UCF-Crime dataset
```bash
python ucf_train.py
python ucf_test.py
```
Traing and infer for XD-Violence dataset
```bash
python xd_train.py
python xd_test.py
```

# References
We used the following repos as references when developing the code.
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
* [DeepMIL](https://github.com/Roc-Ng/DeepMIL)
* [VADCLIP](https://github.com/nwpu-zxr/VadCLIP?tab=readme-ov-file#train-and-test)

