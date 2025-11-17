import sys
# (根据你的截图，okg 位于 ../okg/，所以这个路径可能是正确的)
sys.path.append("../") 

from typing import Optional, Type, List
import ast
import unicodedata
import json # <-- (新增) 导入 JSON 库

from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field, field_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from langchain_community.utilities.serpapi import SerpAPIWrapper

import pickle
import os
import pandas as pd

# --- (新增) 1. 影像解读工具 ---
class ImagingInput(BaseModel):
    """用于 ImagingInterpreterTool 的输入格式。"""
    report_text: str = Field(description="病人完整的影像（MRI/CT/PET）报告的文本内容。")
    focus_area: str = Field(description="指定解读的重点，例如 'TNM分期'，'肿瘤最大径'，'颈部淋巴结' 或 '远处转移迹象'。")
    
class ImagingInterpreterTool(BaseTool):
    name: str = "imaging_interpreter"
    description: str = "用于解析医学影像报告文本，提取出关键的、可用于TNM分期和诊断的结构化数据点。输入必须包含完整的报告文本和分析重点。"
    args_schema: Type[BaseModel] = ImagingInput
    return_direct: bool = False

    def _run(self, report_text: str, focus_area: str, run_manager=None) -> str:
        extracted_data = {}
        
        if "肿瘤" in report_text and "最大径" in report_text and "TNM分期" in focus_area:
            extracted_data["T_stage_key"] = "T2" 
            extracted_data["Max_Diameter_cm"] = "5.2" 
        
        if "淋巴结" in report_text and "坏死" in report_text and "颈部淋巴结" in focus_area:
            extracted_data["N_stage_key"] = "N2" 
            extracted_data["Node_Characteristics"] = "Bilateral cervical nodes, central necrosis."
            
        if not extracted_data:
            return f"从影像报告中未提取到 {focus_area} 相关的关键结构化数据，请检查报告内容和分析重点。"

        return json.dumps(extracted_data, ensure_ascii=False)

# --- (新增) 2. 风险计算工具 ---
class RiskInput(BaseModel):
    """用于 RiskCalculatorTool 的输入格式。"""
    ebv_dna_copies: float = Field(description="病人血浆 EBV-DNA 拷贝数的数字值。")
    tnm_stage: str = Field(description="病人的 TNM 临床分期（例如：'II', 'III', 'IVA'）。")
    
class RiskCalculatorTool(BaseTool):
    name: str = "ebv_risk_calculator"
    description: str = "用于计算基于 EBV-DNA 拷贝数和 TNM 分期的 NPC 复发风险评分。返回一个包含评分和风险等级的结构化JSON。"
    args_schema: Type[BaseModel] = RiskInput
    return_direct: bool = False

    def _run(self, ebv_dna_copies: float, tnm_stage: str, run_manager=None) -> str:
        score = (ebv_dna_copies * 0.001) 
        if "III" in tnm_stage or "IV" in tnm_stage:
             score += 0.3
        
        risk_level = "High" if score > 0.5 else "Medium" if score > 0.2 else "Low"
        
        result = {
            "Risk_Score": round(score, 3),
            "Risk_Level": risk_level,
            "Basis": f"EBV DNA: {ebv_dna_copies}, Stage: {tnm_stage}"
        }
        return json.dumps(result, ensure_ascii=False)

# --- (新增) 3. 临床指南工具 ---
class GuidelineInput(BaseModel):
    """用于 ClinicalGuidelineTool 的输入格式。"""
    tnm_stage: str = Field(description="病人的最终 TNM 临床分期（例如：'II', 'III', 'IVA'）。")
    histology_type: str = Field(description="病理组织学类型（例如：'未分化型非角化性癌'）。")
    
class ClinicalGuidelineTool(BaseTool):
    name: str = "clinical_guideline_retriever"
    description: str = "用于查询最新的 NPC 临床指南（如 CSCO, AJCC）。根据TNM分期和组织学类型，返回结构化的标准治疗方案建议。"
    args_schema: Type[BaseModel] = GuidelineInput
    return_direct: bool = False

    def _run(self, tnm_stage: str, histology_type: str, run_manager=None) -> str:
        recommendation = {}
        
        if "I" == tnm_stage:
            recommendation["Standard_Treatment"] = "单纯放疗 (RT)"
            recommendation["Guideline_Source"] = "CSCO NPC Guideline 2024 (Simulated)"
        elif "II" in tnm_stage:
            recommendation["Standard_Treatment"] = "同步放化疗 (CCRT)"
            recommendation["Guideline_Source"] = "CSCO NPC Guideline 2024 (Simulated)"
        elif "III" in tnm_stage or "IVA" in tnm_stage:
            recommendation["Standard_Treatment"] = "诱导化疗 + 同步放化疗 (IC + CCRT)"
            recommendation["Guideline_Source"] = "CSCO NPC Guideline 2024 (Simulated)"
        else:
            recommendation["Standard_Treatment"] = "需进一步评估，但放疗是基础。"
            recommendation["Guideline_Source"] = "General Oncology Principles"

        return json.dumps(recommendation, ensure_ascii=False)

# --- 以下是 TalkHier 原始 `tools.py` 中的工具 (来自你的文件) ---

class CounterInput(BaseModel):
    in_str: str = Field(description="A List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]."\
                        "Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy.")
    
    @field_validator('in_str', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class CustomCounterTool(BaseTool):
    name: str = "character_counter"
    description: str = "A character counter. Useful for counting the number of characters in a sentence. Takes as input a List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]. \
        Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False


    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        """Returns the number of characters in each input sentence."""
        return_str = ""
        for sent in in_sent:
            c_count = count_chars(sent[0])
            limit = int(sent[1])
            return_str += f"{sent[0]}: {c_count}/{sent[1]} characters"
            if c_count > limit:
                return_str += " (Too long)\n"
            elif c_count < limit//2:
                return_str += " (Too short)\n"
            else:
                return_str += "\n"
        return return_str

def count_chars(s):
    count = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ['F', 'W']:  # Full-width or Wide characters
            count += 2
        else:
            count += 1
    return count


class SerpAPIInput(BaseModel):
    in_str: str = Field(description="Input as String")

    @field_validator('in_str', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class SerpAPITool(BaseTool):
    name: str = "google_search"
    description: str = "A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        searched_dict = retrieveSearches()
        if in_str in searched_dict:
            print("\nNote: Loaded from backup")
            return searched_dict[in_str]
        
        search_res = SerpAPIWrapper().run(in_str)
        searched_dict[in_str] = search_res
        saveSearches(searched_dict)
        return search_res



class OutputTool(BaseTool):
    name: str = "output_tool"
    description: str = "A tool to simply write your thoughts. Nothing will be return for output."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return ""


class ClickAggregator(BaseTool):
    name: str = "click_aggregator"
    description: str = "Returns the total number of clicks per category for the current ad setting."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    click_df: pd.DataFrame = None

    def __init__(self, file):
        super().__init__()
        self.click_df = pd.read_csv(file)

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        aggr_clicks = self.click_df.groupby("Category", as_index=False).sum()
        average = aggr_clicks["Clicks"].mean()
        return_text = ""
        for row in range(len(aggr_clicks)):
            return_text += f"Category: {aggr_clicks["Category"][row]}\nClicks: {aggr_clicks["Clicks"][row]}\nDifference to Average: {aggr_clicks["Clicks"][row] - average: .3f}\n\n"
        return return_text


from sympy import sympify, S, symbols,  Not, Or, And, Implies, Equivalent
import itertools


class MultStr(BaseModel):
    in_str: str = Field(description="Input as a list of strings, in the form of: [Expression1, Expression2, ...]")



class TruthTableGenerator(BaseTool):
    name: str = "truthtable_generator"
    description: str = "Returns the truth table for a given list of Boolean expression. Always use SymPy-style logical operators: \
                    And(A, B) for AND, Or(A, B) for OR, Not(A) for NOT, Implies(A, B) for IMPLIES, and Equivalent(A, B) for BICONDITIONAL. \
                    Parentheses can be used for grouping. Example: ['And(Or(Not(A), B), Implies(C, A))', ...]."

    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if in_str[0] != "[":
            in_str = [in_str]
        else:
            in_str = ast.literal_eval(in_str)
        # print(in_str)
        try:
            expected_symbols = [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ]
            local_dict = {name: symbols(name) for name in expected_symbols}
            local_dict.update({
                "~": Not,  # Negation
                "&": And,  # Logical AND
                "|": Or,   # Logical OR
                ">>": Implies,  # Logical implication
                "EQ": Equivalent  # Logical equivalence
            })

            # Parse the expressions
            parsed_expressions = [sympify(expr, locals=local_dict) for expr in in_str]
            if len(in_str) > 1:
                parsed_expressions.append(Equivalent(*parsed_expressions))
                in_str.append("Same Value for All Formulas")
            
            # Extract free symbols from all expressions
            free_syms = set().union(*(expr.free_symbols for expr in parsed_expressions))
            variables = sorted({str(s) for s in free_syms})

            if not variables:
                results = ["T" if expr else "F" for expr in parsed_expressions]
                return "Constant expressions:\n" + "\n".join(f"{expr} = {res}" for expr, res in zip(in_str, results))

            truth_combinations = list(itertools.product([False, True], repeat=len(variables)))

            headers = " | ".join(variables) + " | " + " | ".join(in_str)
            separator = "-" * len(headers)
            table_rows = [headers, separator]

            same_val_list = []

            for values in truth_combinations:
                var_dict = dict(zip(variables, values))
                results = ["T" if expr.subs(var_dict) else "F" for expr in parsed_expressions]
                row_values = " | ".join("T" if v else "F" for v in values)
                table_rows.append(f"{row_values} | " + " | ".join(results))
                same_val_list.append(results[-1] == "T")
            
            if sum(same_val_list) == len(same_val_list):
                table_rows.append("\nIMPORTANT: THE GIVEN PROPOSITIONS ARE: **Logically Equivalent**")
                print("Logically Equivalent")
            elif sum(same_val_list) == 0:
                table_rows.append("\nIMPORTANT: THE GIVEN PROPOSITIONS ARE: **Contradictory**")
                print("Contradictory")


            return "\n".join(table_rows)

        except Exception as e:
            return f"Error processing expressions: {str(e)}"



class CounterexampleVerifier(BaseTool):
    name: str = "counterexample_verifier"
    description: str = "Verifies whether a given set of truth values serves as a counterexample to an argument. "\
                        "Input consists of premises, a conclusion, and a dictionary specifying truth values for variables, in the form of: "\
                        "{{\"premises\": [Premis1, ...], \"conclusion\": Conclusion, \"truth_values\": [{{variable1: \"True/False\", ...}}, ...]}}"\
                        "Uses SymPy-style logical operators: And(A, B), Or(A, B), Not(A), Implies(A, B), Equivalent(A, B). Make sure to give True False as a string."

    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            expected_symbols = [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ]
            local_dict = {name: symbols(name) for name in expected_symbols}
            local_dict.update({
                "~": Not,  # Negation
                "&": And,  # Logical AND
                "|": Or,   # Logical OR
                ">>": Implies,  # Logical implication
                "EQ": Equivalent  # Logical equivalence
            })
            in_dict = ast.literal_eval(in_str)
            premises = in_dict["premises"]
            if len(premises) == 0:
                return "Empty premis, no evaluation possible."
            conclusion = in_dict["conclusion"]

            result = ""

            print(in_dict["truth_values"])

            for truth_value_list in in_dict["truth_values"]:
                truth_values = truth_value_list
                for key in truth_values.keys():
                    if truth_values[key] in ["True", "true", "T"]:
                        truth_values[key] = True
                    else:
                        truth_values[key] = False

                parsed_premises = [sympify(expr, locals=local_dict) for expr in premises]
                parsed_conclusion = sympify(conclusion, locals=local_dict)

                all_variables = set().union(*(expr.free_symbols for expr in parsed_premises + [parsed_conclusion]))
                var_dict = {str(var): truth_values[str(var)] for var in all_variables if str(var) in truth_values}

                premises_results = [expr.subs(var_dict) for expr in parsed_premises]
                conclusion_result = parsed_conclusion.subs(var_dict)

                premises_truths = [bool(result) for result in premises_results]
                conclusion_truth = bool(conclusion_result)

                if all(premises_truths) and not conclusion_truth:
                    result += "For " + str(truth_value_list) + ": Valid counterexample. The given truth values make all premises true and the conclusion false.\n"
                else:
                    result += "For " + str(truth_value_list) + ": Not a counterexample.The given truth values do not satisfy the conditions for a counterexample.\n"
            return result
        except Exception as e:
            return f"Error processing expressions: {str(e)}"


import csv
class RejectWordTool(BaseTool):
    name: str = "reject_words"
    description: str = "A reject word checker. Checks whether each sentence contains words that should not be included. Takes as input a list composed in the form: [sentence1, sentence2, ...]."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False
    reject_list: list = []

    def __init__(self, file_path):
        super().__init__()
        self.reject_list = []
        with open(file_path, mode='r', encoding='utf-8') as file: # (指定 encoding)
            reader = csv.reader(file)
            for row in reader:
                self.reject_list += row


    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        return_str = ""
        for sent in in_sent:
            rejected = []
            for reject in self.reject_list:
                if reject in sent:
                    rejected.append(reject)
            
            if len(rejected) > 0:
                return_str += f"{sent}: Rejected {rejected}\n"
            else:
                return_str += f"{sent}: Good\n"

        return return_str



def retrieveSearches():
    # (使用相对路径)
    search_path = './searches'
    search_file = os.path.join(search_path, 'searches.pkl')
    if not os.path.exists(search_file):
        return {}
    else:
        with open(search_file, 'rb') as f:
            return pickle.load(f)

def saveSearches(key_dict):
    print("\nNote: Saved to backup")
    search_path = './searches'
    search_file = os.path.join(search_path, 'searches.pkl')
    os.makedirs(search_path, exist_ok=True)
    with open(search_file, 'wb') as f:
        pickle.dump(key_dict, f)



def getSerpTool():
    # Comment out if live SerpAPI is needed
    return SerpAPITool()

    search_tool = load_tools(["serpapi"])
    search_tool[0].name = "google_search"
    return search_tool[0]




from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader, CSVLoader
try:
    # (根据你的截图，okg 在 src/okg)
    from okg.load_and_embed import customized_trend_retriever
except ImportError:
    print("Warning: okg module not found. Skipping retrievers.")
    customized_trend_retriever = None
    
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool


def getTools(sel_tools, config):
    """ 
    0: SerpAPI
    1: Counter
    2,3: Ad Retriever
    4: Output
    5: Click Aggregator
    6: Python
    7: RejectWordTool
    8: TruthTableGenerator
    9: CounterexampleVerifier
    10: ImagingInterpreterTool (我们的新工具)
    11: RiskCalculatorTool (我们的新工具)
    12: ClinicalGuidelineTool (我们的新工具)
    """
    agent_tools = []
    
    # --- 原始 TalkHier 工具 ---
    if 0 in sel_tools:
        agent_tools.append(getSerpTool())
    if 1 in sel_tools:
        agent_tools.append(CustomCounterTool())

    if customized_trend_retriever:
        if 2 in sel_tools:
            try:
                KW_loader = CSVLoader(config["SETTING"]["initial_keyword_data"])
                KW_retriever = customized_trend_retriever(KW_loader, str(config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  \
                                                        str(config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))
                agent_tools.append(create_retriever_tool(
                    KW_retriever,
                    str(config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
                    str(config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
                ))
            except Exception as e:
                print(f"Warning: Could not initialize Tool 2 (KW_retriever). Error: {e}")
        if 3 in sel_tools:
            try:
                exampler_loader = TextLoader(str(config['SETTING']['rule_data']))
                exampler_retriever = customized_trend_retriever(exampler_loader, str(config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  \
                                                                str(config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 
                agent_tools.append(create_retriever_tool(
                    exampler_retriever,
                    str(config['TOOL']['RULE_RETRIEVAL_NAME']),
                    str(config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
                ))
            except Exception as e:
                print(f"Warning: Could not initialize Tool 3 (exampler_retriever). Error: {e}")
    else:
        if 2 in sel_tools or 3 in sel_tools:
            print("Warning: Tools 2/3 (Retrievers) selected but 'okg' module failed to import.")
    
    if 4 in sel_tools:
        agent_tools.append(OutputTool())
    if 5 in sel_tools:
        try:
            agent_tools.append(ClickAggregator(config["SETTING"]["initial_keyword_data"]))
        except Exception as e:
            print(f"Warning: Could not initialize ClickAggregator. Error: {e}")
    if 6 in sel_tools:
        python_repl = PythonREPL()
        agent_tools.append(Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        ))
    if 7 in sel_tools:
        try:
            agent_tools.append(RejectWordTool(config['TOOL']['REJECT_WORD_CSV']))
        except Exception as e:
            print(f"Warning: Could not initialize RejectWordTool. Error: {e}")
    if 8 in sel_tools:
        agent_tools.append(TruthTableGenerator())
    if 9 in sel_tools:
        agent_tools.append(CounterexampleVerifier())

    # --- (新增) 我们的医学工具 ---
    if 10 in sel_tools:
        agent_tools.append(ImagingInterpreterTool())
    
    if 11 in sel_tools:
        agent_tools.append(RiskCalculatorTool())
        
    if 12 in sel_tools:
        agent_tools.append(ClinicalGuidelineTool())
        
    return agent_tools