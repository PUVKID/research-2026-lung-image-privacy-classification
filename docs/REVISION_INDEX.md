# 📚 실제 차수와 원고 단계

연구 단계·원본 폴더 차수·논문 수정·발표 차수는 서로 다른 구분입니다. 실제 파일과 본문으로 확인한 단계만 정리했습니다. 이전 공개 파일과 이름은 보존하며, 현재 원문과 과거 정제 사본을 구별합니다.

| 단계 | 내용 | 파일 종류 | 폴더·자료 | 직전 단계 전체 비교 |
|---|---|---|---|---|
| [연구1 · 하계 EfficientNetV2-S](revisions/study-1.md) | 제58회2026하계학술대회 3쪽 논문에서 512×512 CT의 Blur·Block 단독 변환을 비교한 연구입니다. 원본92.5%, Blur σ32 53.3%, Block32 79.2%, Block1 52.9%는 원고 보고값입니다. 발표와 그래프 JSON·노트북을 함께 읽습니다. 환자 수·세부 분할을 연구2에서 가져와 소급하지 않습니다. | docx·hwp·ipynb·pdf·pptx | [직접 자료](../%EB%85%BC%EB%AC%B8/1%EC%B0%A8_EfficientNetV2-S_%ED%95%98%EA%B3%84_%EC%B5%9C%EC%A2%85.pdf) | [전체 차이](revision-diffs/study-1.diff) |
| [연구2 · ResNet50 후속 원고](revisions/study-2.md) | 19쪽 후속 원고는 ResNet50, 단독20조건·조합100조건, 전체1097장 입력영상 MS-SSIM 기반 점수와 정확도 절충을 다룹니다. 원본98.17%와 Block256+σ0 98.17%·49.65점은 원고 보고값이며 연구1 대비 모델 변경만으로 효과 원인을 단정하지 않습니다. | hwp·ipynb·pdf | [직접 자료](../%EB%85%BC%EB%AC%B8/2%EC%B0%A8_ResNet50_%EC%B5%9C%EC%A2%85.pdf) | [전체 차이](revision-diffs/study-2.diff) |
| [폴더 1차](revisions/folder-1.md) | ResNet50 블록·블러 노트북의 rebuilt·same_settings·metric_matched 변형을 보관한 코드 차수입니다. EfficientNetV2-S의 연구1과 다른 번호 체계입니다. 보조 패치 스크립트와 본인의 로컬 경로도 원문으로 보존합니다. | ipynb·js·py | [직접 자료](../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/1%EC%B0%A8/block_encryption_resnet50_metric_matched_seq.ipynb) | [전체 차이](revision-diffs/folder-1.diff) |
| [폴더 2차](revisions/folder-2.md) | Blur·Block 설정 정렬, 조합 노트북 생성, 원본 복원을 담당한 보조 스크립트의 차수입니다. 스크립트에는 본인 컴퓨터의 절대경로가 내장되어 있으며 원문을 보존합니다. 실행 전 사용자의 실제 경로를 확인해야 합니다. 연구2 논문 번호와 같은 뜻이 아닙니다. | mjs | [직접 자료](../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/2%EC%B0%A8/align_experiment_methods.mjs) | [전체 차이](revision-diffs/folder-2.diff) |
| [폴더 3차](revisions/folder-3.md) | Privacy Score 계산과 소규모 검사를 정리한 자료 차수입니다. 저장 CSV의 n_images=3이고 block2·blur2·combo4조건이므로 논문2의1097장 전체 결과로 소개하지 않습니다. 의료 원본 비교그림은 제외하고 집계4CSV·통합표·비의료 도표9개를 보존합니다. 추가 선택목록의 파일명·본인 경로는 의료영상 바이트와 구별합니다. | csv·ipynb·png·py·xlsx | [직접 자료](../%EC%9E%90%EB%A3%8C%EC%B0%A8%EC%88%98/3%EC%B0%A8/tables/privacy_scores_best_conditions.csv) | [전체 차이](revision-diffs/folder-3.diff) |
| [폴더 4차](revisions/folder-4.md) | ResNet50 후속 논문의 원본 사본·코드·도표와 원고 점검 산출물을 모은 차수입니다. 공개에는 대응 논문과 코드만 연결하며 개인 검토·실행로그와 의료영상 그림은 제외합니다. | hwp·ipynb·pdf | [직접 자료](../%EB%85%BC%EB%AC%B8/2%EC%B0%A8_ResNet50_%EC%B5%9C%EC%A2%85.pdf) | [전체 차이](revision-diffs/folder-4.diff) |
| [블러 · 2차 실험](revisions/experiment-6.md) | 입력 영상에 σ0~256의 Gaussian Blur 10조건을 적용하고 ResNet50으로 분류합니다. 테스트를 먼저 분리한 뒤 학습 자료를 증강합니다. | ipynb·json | [직접 자료](../%EB%B8%94%EB%9F%AC/2%EC%B0%A8%20%EC%8B%A4%ED%97%98/blur_resnet50_rebuilt2_KO.ipynb) | [전체 차이](revision-diffs/experiment-6.diff) |
| [블록 · 2차 실험](revisions/experiment-7.md) | 입력 블록을 512~1 크기로 재배열하는 ResNet50 10조건 실험입니다. 이 사본은 증강 후 테스트를 나눕니다. | ipynb·json | [직접 자료](../%EB%B8%94%EB%A1%9D/2%EC%B0%A8%20%EC%8B%A4%ED%97%98/block_encryption_resnet50_rebuilt_KO.ipynb) | [전체 차이](revision-diffs/experiment-7.diff) |
| [블록+블러 · 기본](revisions/experiment-8.md) | 블록 재배열 뒤 입력 블러를 적용하는 ResNet50 100조합 실험입니다. 조합별 정확도·ROC-AUC·혼동행렬을 저장합니다. | ipynb·json | [직접 자료](../%EB%B8%94%EB%A1%9D%2B%EB%B8%94%EB%9F%AC/block_blur_resnet50_v4_KO.ipynb) | [전체 차이](revision-diffs/experiment-8.diff) |
| [블러 엣지 · 기본](revisions/experiment-9.md) | 입력 블러 10조건에 ResNet50 edge/cloud 분리를 더했습니다. pool1_pool 특징 그림과 부분 모델을 저장하며, 셀 20 문법 오류가 남아 있습니다. | ipynb·json | [직접 자료](../%EB%B8%94%EB%9F%AC%20%EC%97%A3%EC%A7%80/blur_resnet50_edge_v6.ipynb) | [전체 차이](revision-diffs/experiment-9.diff) |
| [블록 엣지 · 기본](revisions/experiment-10.md) | 입력 블록 재배열 10조건을 pool1_pool 경계의 분리 모델로 학습합니다. 블록별 특징 그림과 edge·cloud 가중치를 저장합니다. | ipynb·json | [직접 자료](../%EB%B8%94%EB%A1%9D%20%EC%97%A3%EC%A7%80/block_encryption_resnet50_edge_v2.ipynb) | [전체 차이](revision-diffs/experiment-10.diff) |
| [블록+블러 엣지 · 기본](revisions/experiment-11.md) | 입력 Block→Blur 100조합을 분리 모델로 학습합니다. 특징 자체의 블러 없이 edge/cloud를 같은 프로세스에서 연결합니다. | ipynb | [직접 자료](../%EB%B8%94%EB%A1%9D%2B%EB%B8%94%EB%9F%AC%20%EC%97%A3%EC%A7%80/block_blur_resnet50_edge.ipynb) | [전체 차이](revision-diffs/experiment-11.diff) |
| [블러 스테이지 1 엣지 · 기본](revisions/experiment-12.md) | pool1_pool 특징 자체에 블러를 적용합니다. 특징 SSIM·cosine 및 복원 PSNR/SSIM 코드가 있으며 큰 σ 결과를 재사용합니다. | ipynb·json | [직접 자료](../%EB%B8%94%EB%9F%AC%20%EC%8A%A4%ED%85%8C%EC%9D%B4%EC%A7%80%201%20%EC%97%A3%EC%A7%80/stage1_edge_feature_blur_resnet50.ipynb) | [전체 차이](revision-diffs/experiment-12.diff) |
| [원본 실험 · 기본](revisions/experiment-13.md) | 224×224 조직영상 5라벨 Streamlit 앱과 MobileNetV2 3클래스 학습을 보관합니다. 두 코드의 출력 라벨·모델 파일이 다릅니다. | ipynb·md·png·py·기타 | [직접 자료](../%EC%9B%90%EB%B3%B8%20%EC%8B%A4%ED%97%98/Lung%20Cancer/main.ipynb) | [전체 차이](revision-diffs/experiment-13.diff) |
| [추가 학습모델과 실험 · 기본](revisions/experiment-14.md) | EfficientNetV2-S 변환 비교, MobileNetV2 풀링, ResNet50 사본과 SSIM/MS-SSIM 계산·도표를 모았습니다. 셀별 실험을 구분합니다. | csv·ipynb·json·py·xlsx | [직접 자료](../%EC%B6%94%EA%B0%80%20%ED%95%99%EC%8A%B5%EB%AA%A8%EB%8D%B8%EA%B3%BC%20%EC%8B%A4%ED%97%98/block_blur_resnet50_edge.ipynb) | [전체 차이](revision-diffs/experiment-14.diff) |
| [블러 · 1차 실험](revisions/experiment-original-blur.md) | 원본 폴더의 첫 실험에서 저장한 결과 JSON입니다. 하계 원고의 그래프 입력 자료와 별도로 원본 위치·바이트·해시를 보존합니다. 연구1 번호와 폴더1차 실험은 구별하며 재학습을 수행하지 않았습니다. | json | [직접 자료](../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%EB%B8%94%EB%9F%AC/1%EC%B0%A8%20%EC%8B%A4%ED%97%98/%EB%B8%94%EB%9F%AC%201/%EB%B8%94%EB%9F%AC%EB%A7%81%20%EA%B2%B0%EA%B3%BC/all_blur_results.json) | [전체 차이](revision-diffs/experiment-original-blur.diff) |
| [블록 · 1차 실험](revisions/experiment-original-block.md) | 원본 폴더의 첫 실험에서 저장한 결과 JSON입니다. 하계 원고의 그래프 입력 자료와 별도로 원본 위치·바이트·해시를 보존합니다. 연구1 번호와 폴더1차 실험은 구별하며 재학습을 수행하지 않았습니다. | json | [직접 자료](../%EC%9B%90%EB%B3%B8%EC%9E%90%EB%A3%8C/%EB%B8%94%EB%A1%9D/1%EC%B0%A8%20%EC%8B%A4%ED%97%98/%EB%B8%94%EB%A1%9D%201/%EB%B6%88%EB%A1%9D%20%EA%B2%B0%EA%B3%BC/all_block_results.json) | [전체 차이](revision-diffs/experiment-original-block.diff) |


<details>
<summary>🧹 공개 자료 정리 기록</summary>

| 검수 대상 | 연결 자료 | 상세·이번 비교 |
| :--- | :--- | :--- |
| **1차** | [📁 검수 자료](../1%EC%B0%A8) | [📖 정리 내용](publication-review/item-1.md) · [🔎 파일 비교](publication-review/공개파일변경.json) |

</details>
