import sys
# (不需要 sys.path.append，因为 experiments.py 和 multiagent 在同一 src 目录下)
import configparser
import json
from uuid import uuid4
import argparse # 保留原始库的 argparse
import os # (新增)

# 导入 TalkHier 核心组件
from multiagent.llm import LLM
from multiagent.agent_team import AgentTeam, ReactAgent, buildTeam
from multiagent.react_agent import ReactAgent # 确保 ReactAgent 被导入

# 导入你的 NPC 配置文件 (从 .prompts 目录)
from prompts.medical_diag import get_npc_prompts
# (保留 mmlu 的导入，以防你需要运行原始实验)
from prompts.mmlu import getPrompts as get_mmlu_prompts

# 导入 Langchain/Langgraph 消息类型
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import Sequence

def load_config(config_file="../config_llm.ini"):
    """
    读取 INI 配置文件。
    (注意：从 src/experiments.py 访问根目录的 config_llm.ini，路径是 '../config_llm.ini')
    """
    print(f"Loading config from {config_file}...")
    if not os.path.exists(config_file):
         raise FileNotFoundError(f"Config file not found at {config_file}. Make sure it is in the project root.")
         
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 检查密钥是否存在
    api_key = config['KEYS'].get('OPENAI_API_KEY')
    if not api_key or '***' in api_key:
        raise ValueError("OPENAI_API_KEY is not set or is still a placeholder in config_llm.ini")
        
    return config

def run_npc_consultation(patient_case: dict, llm_instance: LLM, config: configparser.ConfigParser):
    """
    运行一次完整的 NPC MDT 会诊
    """
    print("--- 1. 获取 NPC MDT 团队配置 ---")
    npc_prompt_config = get_npc_prompts(patient_case)
    
    if not isinstance(llm_instance, LLM):
         raise TypeError("llm_instance 必须是 LLM 类的实例")

    # (llm_instance.llm 才是底层的 ChatOpenAI 实例)
    react_generator = ReactAgent(config=config, llm=llm_instance.llm) 

    print("--- 2. 构建 TalkHier MDT 团队工作流 ---")
    team_workflow = buildTeam(
        team_information=npc_prompt_config["team"],
        react_generator=react_generator,
        intermediate_output_desc="结构化的中间输出",
        int_out_format=npc_prompt_config["int_out_format"]
    )
    print("MDT 团队工作流构建完成。")

    print("--- 3. 运行工作流 ---")
    thread_id = f"npc_case_{uuid4()}"
    
    initial_prompt_content = f"开始对病患 {patient_case.get('Patient_ID', 'N/A')} 进行MDT会诊。临床问题：{patient_case.get('Clinical_Question', 'N/A')}"
    
    initial_state = {
        "history": {},
        "intermediate_output": {},
        "background": SystemMessage(content=json.dumps(patient_case, ensure_ascii=False)), 
        "messages": HumanMessage(content=initial_prompt_content),
        "initial_prompt": HumanMessage(content=initial_prompt_content)
    }
    
    print(f"--- <b>正在启动会诊 (Thread ID: {thread_id})</b> ---")
    
    final_output_state = team_workflow.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    print("--- 4. 会诊流程结束 ---")
    
    final_report = final_output_state.get("intermediate_output", {})
    
    return final_report

def main():
    """
    主执行函数 - 替换原始的 main
    """
    # (设置 argparse 来选择运行 MMLU 还是 NPC)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='npc', help='Mode to run: "npc" or "mmlu" (or other original modes)')
    args = parser.parse_args()

    try:
        # 1. 加载配置
        config = load_config()
        
        # 2. 初始化 LLM
        llm = LLM(config) # LLM 类在 init 时加载 llm
        print("LLM 实例加载成功。")

        if args.mode == 'npc':
            # 3. 定义示例病历数据
            example_patient_case = {
                "Patient_ID": "NPC-001",
                "Clinical_Question": "请对该患者的鼻咽癌进行分期和风险评估，并给出初始治疗方案。",
                "MRI_Report_Text": "MRI 报告显示：鼻咽部见一占位病变，侵犯咽旁间隙，最大径 5.2 cm。双侧颈部可见多个 lymph 淋巴结肿大，部分伴中央坏死，最大者 3.0 cm (III区)。",
                "Biopsy_Report_Text": "活检病理：非角化性癌，未分化型。",
                "PET_Report_Text": "PET-CT 显示：除颈部淋巴结外，未见明确远处转移迹象（M0）。",
                "Lab_Data": {
                    "EBV_DNA_Copies": 1500.0, # copies/ml
                    "KPS_Score": 90,
                }
            }

            # 4. 运行会诊
            final_report = run_npc_consultation(example_patient_case, llm, config)
            
            print("\n===================================")
            print("  最终结构化诊断报告 (JSON)  ")
            print("===================================")
            print(json.dumps(final_report, indent=2, ensure_ascii=False))

        else:
            print(f"模式 '{args.mode}' 未被 NPC 启动器处理。请运行原始的 MMLU 逻辑（如果需要）。")
            # (这里你可以保留或调用原始 experiments.py 的 main 函数逻辑)
            # original_main(args, config, llm) 

    except FileNotFoundError as e:
        print(f"错误：文件未找到。{e}")
        print("请确保 `config_llm.ini` 文件位于项目根目录。")
    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()