# 📚 폴더 4차

ResNet50 후속 논문의 원본 사본·코드·도표와 원고 점검 산출물을 모은 차수입니다. 공개에는 대응 논문과 코드만 연결하며 개인 검토·실행로그와 의료영상 그림은 제외합니다.



[전체 차수](../REVISION_INDEX.md) · [첫 화면](../../README.md) · [직전 단계 전체 비교](../revision-diffs/folder-4.diff) · [전체 파일 크기·해시](../revision-manifests/folder-4.json)

직전 단계: 폴더 3차.

## 📁 실제 파일

| 파일 | 종류 | 바이트 | SHA-256 |
|---|---|---:|---|
| [논문/2차_ResNet50_최종.pdf](../../%EB%85%BC%EB%AC%B8/2%EC%B0%A8_ResNet50_%EC%B5%9C%EC%A2%85.pdf) | .pdf | 2827829 | `b64645277f78ac2b2c87a1dcc15216ec9ad6c41a5a20f795465748ac0d2f76c7` |
| [자료차수/4차/코드/block_blur_resnet50_v4_KO.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/block_blur_resnet50_v4_KO.ipynb) | .ipynb | 219524 | `da4b889ce6dc4f4c908cc98986724abd25cd0dbd6aa1425b5e014b1f7a7fee03` |
| [자료차수/4차/코드/block_encryption_resnet50_rebuilt_KO.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/block_encryption_resnet50_rebuilt_KO.ipynb) | .ipynb | 135877 | `cc8eb1485adac09ffdcb2a5c0e502e016f5a5bb1c60898746be7ef9ce87a1986` |
| [자료차수/4차/코드/blur_resnet50_rebuilt2_KO.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/blur_resnet50_rebuilt2_KO.ipynb) | .ipynb | 130270 | `2a3da41eff6a705b2e30668e63d4fe42d4029dc307de2cb197ff8a6434f31faa` |
| [자료차수/4차/코드/heatmaps_windows.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/heatmaps_windows.ipynb) | .ipynb | 8646 | `75c45970b92c8957ea61de8c236e045c3a0d77ed0de942815d3b16ae6e24ef4a` |
| [자료차수/4차/코드/paper_figures_resnet50.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/paper_figures_resnet50.ipynb) | .ipynb | 37162 | `f317a502c0658e3dbcb90cb7af67db526d6594f733b0a0bccb6ec536a3df53c0` |
| [자료차수/4차/코드/pareto_scatter.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/pareto_scatter.ipynb) | .ipynb | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| [자료차수/4차/코드/picture2x2.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/picture2x2.ipynb) | .ipynb | 7054 | `d56ae7885ab9219f62bb695fe1e5de081c7e667ebbf8c5916b87909bff4b2046` |
| [자료차수/4차/코드/plot_privacy_metrics_split.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/plot_privacy_metrics_split.ipynb) | .ipynb | 18984 | `77cc7650ee472043999937da35fd6ac817330ce45a796c584330571ccf1e05be` |
| [자료차수/4차/코드/privacy_score_ssim_psnr_updated.ipynb](../../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/4%EC%B0%A8/%EC%BD%94%EB%93%9C/privacy_score_ssim_psnr_updated.ipynb) | .ipynb | 94241 | `1b25934ba5fbee1a8fa025596bfa852c6cfe2667f9edb56a5fe57852d35ea6b5` |
| [원본자료/4차/work/source/게재논문(한글논문)양식.hwp](../../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/4%EC%B0%A8/work/source/%EA%B2%8C%EC%9E%AC%EB%85%BC%EB%AC%B8%28%ED%95%9C%EA%B8%80%EB%85%BC%EB%AC%B8%29%EC%96%91%EC%8B%9D.hwp.md) | .hwp | 414720 | `7f3d880b8df41d7549973470476f6011ce80ca835f36f7ac62e22acdb978f196` |

## 🔎 해석·검증 범위

원고는 원고 보고값, 저장된 표는 당시 결과 파일, 노트북은 구현 자료입니다. 이번 정리는 재학습·임상 실험·제조 장비 검사·추천 성능 재현을 수행하지 않았습니다. 파일 비교와 PDF 내용 확인은 실제 기능 실행의 성공을 대신하지 않습니다. 원본의 알려진 오류와 0바이트 자료는 보존하며 성공으로 보정하지 않습니다.
