# src/utils/filter_pages_with_answers.py
import os, re, json, shutil
from pathlib import Path

_RE_CAU = re.compile(r'^\s*Câu\s*\d+\b')
_RE_ANS = re.compile(r'^\s*[A-D]\s*[\.\)\-\:]?')

def count_questions_in_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if _RE_CAU.match(line))

def count_answers_in_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if _RE_ANS.match(line))

def starts_with_cau(path) -> bool:
    """SỬA XONG: Chỉ trả True nếu DÒNG ĐẦU TIÊN có chữ là Câu"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            return bool(_RE_CAU.match(line))
    return False

def analyze_pair(q_path: Path, a_path: Path):
    return {
        "q_file": q_path.name,
        "q_count": count_questions_in_file(q_path),
        "a_count": count_answers_in_file(a_path),
        "starts_with_cau": starts_with_cau(q_path)
    }

# ====================== PHẦN DƯỚI GIỮ NGUYÊN ======================
def filter_pages_with_answers(qf="data/questions", af="data/answers", out="data/output"):
    os.makedirs(out, exist_ok=True)
    def sub(): return {"files":[],"count":0}
    summary = {
        "TH1_Q_eq_A": {"bat_dau_Cau": sub(), "khong_bat_dau_Cau": sub(), "count":0},
        "TH2_Q_eq_A_minus_1": {"bat_dau_Cau": sub(), "khong_bat_dau_Cau": sub(), "count":0},
        "TH3_Q_less_than_A_minus_2": {"bat_dau_Cau": sub(), "khong_bat_dau_Cau": sub(), "count":0},
        "TH4_Q_greater_than_A": {"bat_dau_Cau": sub(), "khong_bat_dau_Cau": sub(), "count":0},
        "total_files_processed": 0,
        "files_detail": {}
    }

    q_files = {p.name:p for p in Path(qf).glob("*.txt")}
    a_files = {p.name:p for p in Path(af).glob("*.txt")}
    for name in sorted(set(q_files)&set(a_files)):
        info = analyze_pair(q_files[name], a_files[name])
        Q, A, start = info["q_count"], info["a_count"], info["starts_with_cau"]

        if Q == A:           case = "TH1_Q_eq_A"
        elif Q == A-1:       case = "TH2_Q_eq_A_minus_1"
        elif Q <= A-2:       case = "TH3_Q_less_than_A_minus_2"
        else:                case = "TH4_Q_greater_than_A"

        sub = "bat_dau_Cau" if start else "khong_bat_dau_Cau"
        summary[case][sub]["files"].append(name)
        summary[case][sub]["count"] += 1
        summary[case]["count"] += 1
        summary["total_files_processed"] += 1
        summary["files_detail"][name] = {"Q":Q,"A":A,"case":case,"starts_with_cau":start}

        shutil.copy(q_files[name], Path(out)/name)
        print(f"{name}: Q={Q}, A={A} → {case}.{1 if start else 2}")

    json_path = Path(out)/"page_comparison_summary.json"
    json.dump(summary, open(json_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nHOÀN TẤT! → {json_path}")

if __name__ == "__main__":
    filter_pages_with_answers()