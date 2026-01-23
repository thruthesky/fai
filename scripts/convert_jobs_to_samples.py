#!/usr/bin/env python3
"""
jobs-sg.json → samples.txt 변환 스크립트

싱가포르 구인 정보 JSON 파일을 JAI 학습용 샘플 데이터로 변환합니다.
기존 samples.txt 파일이 있으면 그 뒤에 추가합니다.

사용법:
    uv run python scripts/convert_jobs_to_samples.py
"""

import json
import random
from pathlib import Path


# 질문 유형 목록 (학습 다양성을 위해 랜덤 선택)
QUESTIONS = [
    "이 연락처 정보를 이해하기 쉽게 요약해줘.",
    "체크리스트와 연락처 중심으로 정리해줘.",
    "핵심 연락처와 주의사항을 정리해줘.",
    "연락처가 있다면 함께 정리해줘.",
    "이 구인 정보를 요약해줘.",
    "이 채용 공고의 핵심 정보를 정리해줘.",
]


def format_doc(job: dict) -> str:
    """
    구인 정보를 [DOC] 섹션 형식으로 변환

    Args:
        job: 구인 정보 딕셔너리

    Returns:
        [DOC] 섹션에 들어갈 텍스트
    """
    meta = job.get("meta", {})

    # 기본 정보 구성
    lines = []
    lines.append(f"=== {job.get('title', '채용 공고')} ===")
    lines.append("")

    # 회사 정보
    if meta.get("company_name"):
        lines.append(f"회사명: {meta['company_name']}")
    if meta.get("position"):
        lines.append(f"포지션: {meta['position']}")
    if meta.get("industry"):
        lines.append(f"업종: {meta['industry']}")
    if meta.get("location"):
        lines.append(f"위치: {meta['location']}")
    if meta.get("employment_type"):
        lines.append(f"고용형태: {meta['employment_type']}")

    # 급여 정보
    salary_parts = []
    if meta.get("salary"):
        salary_parts.append(meta["salary"])
    if meta.get("salary_currency"):
        salary_parts.append(f"({meta['salary_currency']})")
    if salary_parts:
        lines.append(f"급여: {' '.join(salary_parts)}")

    # 경력/학력 요건
    if meta.get("experience_level"):
        lines.append(f"경력: {meta['experience_level']}")
    if meta.get("education_level"):
        lines.append(f"학력: {meta['education_level']}")

    # 근무 조건
    if meta.get("remote_work"):
        lines.append(f"원격근무: {meta['remote_work']}")
    if meta.get("deadline"):
        lines.append(f"마감일: {meta['deadline']}")

    lines.append("")

    # 업무 내용
    if meta.get("responsibilities"):
        lines.append("업무 내용:")
        lines.append(meta["responsibilities"])
        lines.append("")

    # 자격 요건
    if meta.get("requirements"):
        lines.append("자격 요건:")
        lines.append(meta["requirements"])
        lines.append("")

    # 필요 기술
    if meta.get("skills"):
        lines.append(f"필요 기술: {meta['skills']}")
        lines.append("")

    # 연락처 정보
    if meta.get("application_url"):
        lines.append(f"WEB: {meta['application_url']}")
    if meta.get("application_email") and meta["application_email"] != "000":
        lines.append(f"EMAIL: {meta['application_email']}")
    if meta.get("contact_phone") and meta["contact_phone"] != "000":
        lines.append(f"TEL: {meta['contact_phone']}")

    return "\n".join(lines)


def format_answer(job: dict) -> str:
    """
    구인 정보를 [ANSWER] 섹션 형식으로 변환

    Args:
        job: 구인 정보 딕셔너리

    Returns:
        [ANSWER] 섹션에 들어갈 텍스트
    """
    meta = job.get("meta", {})

    lines = []

    # 요약 섹션
    lines.append("요약:")
    if meta.get("company_name"):
        lines.append(f"- 회사: {meta['company_name']}")
    if meta.get("position"):
        lines.append(f"- 포지션: {meta['position']}")
    if meta.get("location"):
        lines.append(f"- 위치: {meta['location']}")
    if meta.get("employment_type"):
        lines.append(f"- 고용형태: {meta['employment_type']}")
    if meta.get("salary"):
        salary = meta["salary"]
        if meta.get("salary_currency"):
            salary += f" ({meta['salary_currency']})"
        lines.append(f"- 급여: {salary}")
    if meta.get("deadline"):
        lines.append(f"- 마감일: {meta['deadline']}")

    lines.append("")

    # 체크리스트 섹션
    lines.append("체크리스트:")
    lines.append("- 해야 할 일:")
    lines.append("  - (1) 이력서 준비")
    lines.append("  - (2) 지원서 작성 및 제출")
    if meta.get("skills"):
        lines.append("  - (3) 필요 기술 확인")
    lines.append("- 준비물:")
    lines.append("  - (1) 이력서/CV")
    if meta.get("requirements") and "포트폴리오" in meta["requirements"]:
        lines.append("  - (2) 포트폴리오")
    lines.append("- 주의사항:")
    if meta.get("deadline"):
        lines.append(f"  - (1) 마감일({meta['deadline']}) 확인 필수")
    else:
        lines.append("  - (1) 마감일 확인 필수")
    if meta.get("requirements"):
        lines.append("  - (2) 자격 요건 충족 여부 확인")

    lines.append("")

    # 연락처 섹션
    lines.append("연락처(공공정보):")
    if meta.get("application_url"):
        lines.append(f"- WEB: {meta['application_url']}")
    if meta.get("application_email") and meta["application_email"] != "000":
        lines.append(f"- EMAIL: {meta['application_email']}")
    if meta.get("contact_phone") and meta["contact_phone"] != "000":
        lines.append(f"- TEL: {meta['contact_phone']}")

    lines.append("")

    # 상세 설명 섹션
    lines.append("상세 설명:")
    lines.append("이 문서는 구인 정보를 담고 있습니다.")

    # 담당 업무 요약
    if meta.get("responsibilities"):
        resp = meta["responsibilities"][:100]  # 처음 100자
        if len(meta["responsibilities"]) > 100:
            resp += "..."
        lines.append(f"주요 업무: {resp}")

    # 자격 요건 요약
    if meta.get("requirements"):
        req = meta["requirements"][:100]  # 처음 100자
        if len(meta["requirements"]) > 100:
            req += "..."
        lines.append(f"자격 요건: {req}")

    lines.append("연락처 정보는 변경될 수 있으므로 공식 웹사이트에서 최신 정보를 확인하는 것이 좋습니다.")

    return "\n".join(lines)


def convert_job_to_sample(job: dict) -> str:
    """
    단일 구인 정보를 샘플 데이터 형식으로 변환

    Args:
        job: 구인 정보 딕셔너리

    Returns:
        [QUESTION][DOC][ANSWER] 형식의 샘플 텍스트
    """
    question = random.choice(QUESTIONS)
    doc = format_doc(job)
    answer = format_answer(job)

    sample = f"""[QUESTION]
{question}
[/QUESTION]

[DOC]
{doc}
[/DOC]

[ANSWER]
{answer}
[/ANSWER]
"""
    return sample


def main():
    """메인 실행 함수"""
    # 경로 설정
    project_root = Path(__file__).parent.parent
    json_path = project_root / "data" / "jobs-sg.json"
    samples_path = project_root / "data" / "samples.txt"

    # JSON 파일 읽기
    print(f"JSON 파일 읽는 중: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    print(f"총 {len(jobs)}개의 구인 정보를 발견했습니다.")

    # 기존 파일이 있으면 덮어쓰기 알림
    if samples_path.exists():
        print(f"기존 samples.txt 파일을 덮어씁니다.")

    # 각 구인 정보를 샘플로 변환
    new_samples = []
    for i, job in enumerate(jobs):
        sample = convert_job_to_sample(job)
        new_samples.append(sample)

        # 진행 상황 출력 (100개마다)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(jobs)} 변환 완료...")

    # samples.txt에 저장 (덮어쓰기)
    print(f"\nsamples.txt에 저장 중: {samples_path}")
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in new_samples:
            f.write(sample)
            f.write("\n")

    # 결과 통계
    with open(samples_path, "r", encoding="utf-8") as f:
        final_content = f.read()

    print(f"\n변환 완료!")
    print(f"  - 변환된 구인 정보: {len(jobs)}개")
    print(f"  - 최종 파일 크기: {len(final_content):,} 바이트")
    print(f"  - 저장 위치: {samples_path}")


if __name__ == "__main__":
    main()
