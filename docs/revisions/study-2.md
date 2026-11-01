# 📚 연구2 · ResNet50 후속 원고

19쪽 후속 원고는 ResNet50, 단독20조건·조합100조건, 전체1097장 입력영상 MS-SSIM 기반 점수와 정확도 절충을 다룹니다. 원본98.17%와 Block256+σ0 98.17%·49.65점은 원고 보고값이며 연구1 대비 모델 변경만으로 효과 원인을 단정하지 않습니다.



[전체 차수](../REVISION_INDEX.md) · [첫 화면](../../README.md) · [직전 단계 전체 비교](../revision-diffs/study-2.diff) · [전체 파일 크기·해시](../revision-manifests/study-2.json)

직전 단계: 연구1 · 하계 EfficientNetV2-S.

## 📁 실제 파일

| 파일 | 종류 | 바이트 | SHA-256 |
|---|---|---:|---|
| [논문/2차_ResNet50_최종.pdf](../../%EB%85%BC%EB%AC%B8/2%EC%B0%A8_ResNet50_%EC%B5%9C%EC%A2%85.pdf) | .pdf | 2827829 | `b64645277f78ac2b2c87a1dcc15216ec9ad6c41a5a20f795465748ac0d2f76c7` |
| [폐암 2차 논문/코드/block_blur_resnet50_v4_KO.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/block_blur_resnet50_v4_KO.ipynb) | .ipynb | 136449 | `efdb4bc4e54aef87ca99b2acca4bd36c4f49093351d613ca254c6ddf90de46fc` |
| [폐암 2차 논문/코드/block_encryption_resnet50_rebuilt_KO.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/block_encryption_resnet50_rebuilt_KO.ipynb) | .ipynb | 108653 | `3daed3071f6399ccc1edff47b8925bf31d39ee9a74fc9478f3ec6bee773393af` |
| [폐암 2차 논문/코드/blur_resnet50_rebuilt2_KO.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/blur_resnet50_rebuilt2_KO.ipynb) | .ipynb | 102127 | `745bb2b539259819784a219ebde96df31797fad820af09cd183177e529986870` |
| [폐암 2차 논문/코드/heatmaps_windows.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/heatmaps_windows.ipynb) | .ipynb | 8407 | `e64d162310d50c96a5ba1e39c44fce4702fad38abbf7c08c7a8eaf7b79582590` |
| [폐암 2차 논문/코드/paper_figures_resnet50.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/paper_figures_resnet50.ipynb) | .ipynb | 27771 | `4e52342e6d8a025549a37bf1e16fbd94f2d2a76d95cc0e039e703d6d1fc911ce` |
| [폐암 2차 논문/코드/picture2x2.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/picture2x2.ipynb) | .ipynb | 6594 | `d6ddecbbd7dfd747b83fefe3be920bef1a0597ae3413de936eb7cb27fe4cceb5` |
| [폐암 2차 논문/코드/plot_privacy_metrics_split.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/plot_privacy_metrics_split.ipynb) | .ipynb | 18339 | `6341212d374e771b81529b92f66f672ad66eac891240932b8b49a1c79ef98b0a` |
| [폐암 2차 논문/코드/privacy_score_ssim_psnr_updated.ipynb](../../%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/privacy_score_ssim_psnr_updated.ipynb) | .ipynb | 52527 | `e0ce3aa6dd0a1fa307eea53cb1a2253266431a2ab0d86b984d72010aafc2c6ef` |
| [논문/연구2_원문_최종.pdf](../../%EB%85%BC%EB%AC%B8/%EC%97%B0%EA%B5%AC2_%EC%9B%90%EB%AC%B8_%EC%B5%9C%EC%A2%85.pdf) | .pdf | 2865864 | `58249752c657a4aad13b94d7860bb2d977e897e779e72a9f9e8d6306a60652fe` |
| [논문/연구2_원문_최종.hwp](../../%EB%85%BC%EB%AC%B8/%EC%97%B0%EA%B5%AC2_%EC%9B%90%EB%AC%B8_%EC%B5%9C%EC%A2%85.hwp.md) | .hwp | 4908032 | `14d42c098f5ea8c057873e96136bf597b8f9bcec2c59b563bb763bca96fa1a50` |
| [원본자료/폐암 2차 논문/코드/block_encryption_resnet50_rebuilt_KO.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/block_encryption_resnet50_rebuilt_KO.ipynb) | .ipynb | 138317 | `4233928cb2aba1e9e1306e02ffc8a7cf5f17e62b72b83ccc2e8ec228c592c80f` |
| [원본자료/폐암 2차 논문/코드/blur_resnet50_rebuilt2_KO.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/blur_resnet50_rebuilt2_KO.ipynb) | .ipynb | 132788 | `90c8e0f7a25c4533dc2b6759fa295bc7d758dd256330367d7c395dff2bee0d4f` |
| [원본자료/폐암 2차 논문/코드/heatmaps_windows.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/heatmaps_windows.ipynb) | .ipynb | 8646 | `75c45970b92c8957ea61de8c236e045c3a0d77ed0de942815d3b16ae6e24ef4a` |
| [원본자료/폐암 2차 논문/코드/paper_figures_resnet50.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/paper_figures_resnet50.ipynb) | .ipynb | 38070 | `162e3a670d2aa4dcb74771e651a1fd8c79fdf1e6e0ba33bb4fe81bc1177cc43d` |
| [원본자료/폐암 2차 논문/코드/pareto_scatter.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/pareto_scatter.ipynb) | .ipynb | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| [원본자료/폐암 2차 논문/코드/picture2x2.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/picture2x2.ipynb) | .ipynb | 7054 | `d56ae7885ab9219f62bb695fe1e5de081c7e667ebbf8c5916b87909bff4b2046` |
| [원본자료/폐암 2차 논문/코드/plot_privacy_metrics_split.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/plot_privacy_metrics_split.ipynb) | .ipynb | 18984 | `77cc7650ee472043999937da35fd6ac817330ce45a796c584330571ccf1e05be` |
| [원본자료/폐암 2차 논문/코드/privacy_score_ssim_psnr_updated.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/privacy_score_ssim_psnr_updated.ipynb) | .ipynb | 96269 | `74cb8eb330d2a60c357ff4806f3d4a5c3aeefb3f5f3cc2c0b188d792decb863a` |
| [원본자료/추가 학습모델과 실험/block_blur_resnet50_v4_KO.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%EC%B6%94%EA%B0%80%20%ED%95%99%EC%8A%B5%EB%AA%A8%EB%8D%B8%EA%B3%BC%20%EC%8B%A4%ED%97%98/block_blur_resnet50_v4_KO.ipynb) | .ipynb | 223652 | `988bdd6a86fac7245cdd255f19cdc5bf046579c6bdbe52e95ba8ed96a4f04140` |
| [원본자료/폐암 2차 논문/게재논문(한글논문)양식.hwp](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EA%B2%8C%EC%9E%AC%EB%85%BC%EB%AC%B8%28%ED%95%9C%EA%B8%80%EB%85%BC%EB%AC%B8%29%EC%96%91%EC%8B%9D.hwp.md) | .hwp | 414720 | `7f3d880b8df41d7549973470476f6011ce80ca835f36f7ac62e22acdb978f196` |
| [원본자료/폐암 2차 논문/코드/block_blur_resnet50_v4_KO.ipynb](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%ED%8F%90%EC%95%94%202%EC%B0%A8%20%EB%85%BC%EB%AC%B8/%EC%BD%94%EB%93%9C/block_blur_resnet50_v4_KO.ipynb) | .ipynb | 223652 | `988bdd6a86fac7245cdd255f19cdc5bf046579c6bdbe52e95ba8ed96a4f04140` |

## 🔎 해석·검증 범위

원고는 원고 보고값, 저장된 표는 당시 결과 파일, 노트북은 구현 자료입니다. 이번 정리는 재학습·임상 실험·제조 장비 검사·추천 성능 재현을 수행하지 않았습니다. 파일 비교와 PDF 내용 확인은 실제 기능 실행의 성공을 대신하지 않습니다. 원본의 알려진 오류와 0바이트 자료는 보존하며 성공으로 보정하지 않습니다.
