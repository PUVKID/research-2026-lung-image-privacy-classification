# 📚 연구1 · 하계 EfficientNetV2-S

제58회2026하계학술대회 3쪽 논문에서 512×512 CT의 Blur·Block 단독 변환을 비교한 연구입니다. 원본92.5%, Blur σ32 53.3%, Block32 79.2%, Block1 52.9%는 원고 보고값입니다. 발표와 그래프 JSON·노트북을 함께 읽습니다. 환자 수·세부 분할을 연구2에서 가져와 소급하지 않습니다.



[전체 차수](../REVISION_INDEX.md) · [첫 화면](../../README.md) · [직전 단계 전체 비교](../revision-diffs/study-1.diff) · [전체 파일 크기·해시](../revision-manifests/study-1.json)

직전 단계: 확인된 비교 대상 없음.

## 📁 실제 파일

| 파일 | 종류 | 바이트 | SHA-256 |
|---|---|---:|---|
| [논문/1차_EfficientNetV2-S_하계_최종.pdf](../../%EB%85%BC%EB%AC%B8/1%EC%B0%A8_EfficientNetV2-S_%ED%95%98%EA%B3%84_%EC%B5%9C%EC%A2%85.pdf) | .pdf | 410164 | `dd6309e6f6b722d6db242e9c2237584314ad7d7bdf32eb1a91d10ced5ba3fe63` |
| [연구1_하계_EfficientNetV2-S/그래프/Blur_Block_1차_그래프_white.ipynb](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EA%B7%B8%EB%9E%98%ED%94%84/Blur_Block_1%EC%B0%A8_%EA%B7%B8%EB%9E%98%ED%94%84_white.ipynb) | .ipynb | 31695 | `a8c2615ce407a6e49de77cbea440d236e423dd77439abeaca57697b21e928823` |
| [연구1_하계_EfficientNetV2-S/발표자료/하계학술대회 PPT.pdf](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C%20PPT.pdf) | .pdf | 2156389 | `22c41af0ada654bbaf8c8c5882ca1165da5e1216430af199db1fd0f560f349b3` |
| [연구1_하계_EfficientNetV2-S/발표자료/하계학술대회 PPT.pptx](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C%20PPT.pptx) | .pptx | 2325109 | `a95905aea2ed35a70f0e43e5f0eaf30cbb5c227fed7384f952641104b26d5193` |
| [연구1_하계_EfficientNetV2-S/발표자료/하계학술대회 발표 대본 정리.docx](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C%20%EB%B0%9C%ED%91%9C%20%EB%8C%80%EB%B3%B8%20%EC%A0%95%EB%A6%AC.docx) | .docx | 12063 | `fbe895f010f6ee875395b5486b42dcf7d07248e6b6ac98a17a21bb38eda67246` |
| [연구1_하계_EfficientNetV2-S/발표자료/하계학술대회 발표 대본 정리.pdf](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C%20%EB%B0%9C%ED%91%9C%20%EB%8C%80%EB%B3%B8%20%EC%A0%95%EB%A6%AC.pdf) | .pdf | 280123 | `0dad95908a9f7771703c37c882b5265162b1ea6d1a75762a96cef4d30923a598` |
| [연구1_하계_EfficientNetV2-S/발표자료/하계학술대회 발표 대본.docx](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C%20%EB%B0%9C%ED%91%9C%20%EB%8C%80%EB%B3%B8.docx) | .docx | 14644 | `3a66466626c76c84640e1d4284ff375708f7ba428d92fd9f7a2c686cf3bbbae8` |
| [연구1_하계_EfficientNetV2-S/발표자료/하계학술대회 발표 대본.pdf](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C%20%EB%B0%9C%ED%91%9C%20%EB%8C%80%EB%B3%B8.pdf) | .pdf | 314134 | `f7d5589105d6a714a62ebe53a264a23070941e4fd26db78ddc8a08e8efc2f62a` |
| [연구1_하계_EfficientNetV2-S/논문/원문_최종.hwp](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EB%85%BC%EB%AC%B8/%EC%9B%90%EB%AC%B8_%EC%B5%9C%EC%A2%85.hwp.md) | .hwp | 527360 | `f19ddfe212b9c38fd1cfdfec29f8d499e0150bd183e2d6a7992867b34720b7fc` |
| [연구1_하계_EfficientNetV2-S/원문보관/하계학술대회_논문_심사용.hwp](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EC%9B%90%EB%AC%B8%EB%B3%B4%EA%B4%80/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C_%EB%85%BC%EB%AC%B8_%EC%8B%AC%EC%82%AC%EC%9A%A9.hwp.md) | .hwp | 534016 | `a50590b82b4f2a37c948d267c03acdb38e2dfdb6ec1b912ad77718c50d8b9915` |
| [연구1_하계_EfficientNetV2-S/원문보관/하계학술대회_초록_심사용.hwp](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EC%9B%90%EB%AC%B8%EB%B3%B4%EA%B4%80/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C_%EC%B4%88%EB%A1%9D_%EC%8B%AC%EC%82%AC%EC%9A%A9.hwp.md) | .hwp | 50688 | `218bd80c6e58efc061d4717f1e47dab5c494c748813b3e580d98660a54cb8a4b` |
| [연구1_하계_EfficientNetV2-S/원문보관/하계학술대회_초록_최종본.hwp](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EC%9B%90%EB%AC%B8%EB%B3%B4%EA%B4%80/%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C_%EC%B4%88%EB%A1%9D_%EC%B5%9C%EC%A2%85%EB%B3%B8.hwp.md) | .hwp | 55296 | `d8d43b6b577716c444a38c1fe258184834317b69628d469d075bc1014ba21d44` |
| [연구1_하계_EfficientNetV2-S/원문보관/그림/Blur_Block_1차_그래프_white.ipynb](../../%EC%97%B0%EA%B5%AC1_%ED%95%98%EA%B3%84_EfficientNetV2-S/%EC%9B%90%EB%AC%B8%EB%B3%B4%EA%B4%80/%EA%B7%B8%EB%A6%BC/Blur_Block_1%EC%B0%A8_%EA%B7%B8%EB%9E%98%ED%94%84_white.ipynb) | .ipynb | 34700 | `21207ffb5200d44de6a06b908ce7171648cf3d4a2c390c834162e357d61461ec` |

## 🔎 해석·검증 범위

원고는 원고 보고값, 저장된 표는 당시 결과 파일, 노트북은 구현 자료입니다. 이번 정리는 재학습·임상 실험·제조 장비 검사·추천 성능 재현을 수행하지 않았습니다. 파일 비교와 PDF 내용 확인은 실제 기능 실행의 성공을 대신하지 않습니다. 원본의 알려진 오류와 0바이트 자료는 보존하며 성공으로 보정하지 않습니다.
