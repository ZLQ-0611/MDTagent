import json
import ast
from typing import Dict, Any

def get_npc_prompts(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    为 NPC 多学科会诊 (MDT) 团队构建 TalkHier 配置。
    
    Args:
        problem (Dict[str, Any]): 包含原始病历数据的字典。
    
    Returns:
        Dict[str, Any]: 一个包含所有团队结构和 Prompt 的配置字典。
    """

    # --- 1. 定义最终报告的 Pydantic Schema 结构 ---
    FINAL_DIAGNOSIS_SCHEMA = {
        "FinalDiagnosis": "最终诊断结论（例如：NPC Stage III，未分化型）",
        "CertaintyScore": "诊断信心指数 (0.0 - 1.0)",
        "TNM_Staging_Summary": "整合所有证据后的最终TNM分期",
        "ReasoningChain": [
            {
                "Expert_Role": "Specialist/Analyst Role (e.g., MRI_Analyst)",
                "Evidence_Data": "Key data point/Finding (e.g., 'Largest tumor diameter is 5cm')",
                "Conclusion": "Conclusion based on this data (e.g., 'This confirms T2 staging')",
                "Reference": "Clinical Guideline/Tool Used (e.g., imaging_interpreter)"
            }
        ],
        "RecommendedTreatment": "初始建议治疗方案"
    }

    # 将 Schema 转换为转义字符串，供 LLM 结构化输出使用
    int_out_format = json.dumps(FINAL_DIAGNOSIS_SCHEMA).replace('"', '\\"')

    # --- 2. 初始化核心配置字典 ---
    prompt = {}
    
    prompt["problem"] = problem # 存储原始问题数据

    # --- 3. 定义 Level 1 & Level 2 团队结构和路由指令 ---
    prompt["team"] = {
        "team": "Consultation_Team",
        "return": "FINISH", # 顶层团队完成任务后结束
        "prompt": "你是一个 NPC 多学科会诊 (MDT) 的总负责医师。你的任务是系统性地引导诊断流程。",
        
        # Level 1 主管的路由逻辑 (总负责医师)
        "additional_prompt": """
你是一个 NPC 多学科会诊 (MDT) 的总负责医师。你的任务是根据收到的病历，系统性地引导诊断流程。

1. 任务分解路由：
- 流程必须是 **串行** 的：首先，将任务分配给 `Imaging_Team_Supervisor` 完成分期。
- 其次，将任务分配给 `Pathology_Team_Supervisor` 完成确诊和风险分析，同时整合影像团队的结果。
- 再次，将任务分配给 `Treatment_Team_Supervisor` 进行治疗方案推荐。

2. 证据汇集：
- 每当你收到一个专科团队的结构化报告时，你必须将这些报告全部累积在 `intermediate_output` 中，以确保 `Final_Revisor` 可以查看完整的证据链。

3. 最终报告生成：
- 当且仅当你收到所有三个专科团队的结构化报告后，你才能将所有累积的证据和原始病历一起，分配给 `Final_Revisor`，指令其生成最终的诊断报告。
""",

        # --- Level 2: 影像诊断团队配置 ---
        "Imaging_Team": {
            "team": "Imaging_Team",
            "return": "Consultation_Team Supervisor", # 完成后汇报给总负责医生
            "prompt": "你是一个专业的影像诊断团队，负责所有影像数据的分析。",
            "additional_prompt": """
A. 路由决策：你必须判断总负责医师的请求需要哪些数据。
- 如果请求中明确提及“肿瘤分期”、“局部侵犯”或“软组织评估”，请优先将任务分配给 `MRI_Analyst`。
- 如果请求中明确提及“远处转移”、“代谢活性”或“全身筛查”，请优先将任务分配给 `PET_Analyst`。
- 如果请求是全面的分期评估，请依次先分配给 `MRI_Analyst`，然后将结果传递给 `PET_Analyst`。
B. 证据整合：
- 你的任务是将分析员的所有发现整合成一份统一的、结构化的专科报告。
- 你的专科报告必须包含清晰的TNM分期建议（基于你收集的证据）和任何诊断上的不确定性。
- 你的输出（intermediate_output）必须能被 Pathology_Team Supervisor 直接使用。
""",
            # Level 3 成员配置 (见下文填充)
            "MRI_Analyst": {}, 
            "PET_Analyst": {}
        },
        
        # --- Level 2: 病理与生物标记物团队配置 ---
        "Pathology_Team": {
            "team": "Pathology_Team",
            "return": "Consultation_Team Supervisor",
            "prompt": "你是一个病理与生物标记物团队的主管。你的任务是确诊、分型和预后指标的分析，并整合影像团队的结果。",
            "additional_prompt": """
A. 路由决策：在任何诊断开始前，你必须保证活检确诊已完成。
- 首先将任务分配给 `Biopsy_Interpreter` 进行组织学确诊。
- 在获得确诊结果后，将任务分配给 `EBV_Marker_Analyst` 进行生物标记物分析。
B. 证据整合：
- 你的核心任务是整合影像团队的TNM分期（已从上层Supervisor传递给你）和病理团队的组织学诊断与EBV数据。
- 最终的专科报告必须清晰地包含：确诊的组织学类型、肿瘤分级、EBV 拷贝数分析结果，以及基于这些数据的预后风险评估。
""",
            "Biopsy_Interpreter": {},
            "EBV_Marker_Analyst": {}
        },
        
        # --- Level 2: 治疗与预后评估团队配置 ---
        "Treatment_Team": {
            "team": "Treatment_Team",
            "return": "Consultation_Team Supervisor",
            "prompt": "你是一个治疗与预后评估团队的主管。你负责根据最终确诊的分期和指南，推荐最适合的初始治疗方案。",
            "additional_prompt": """
A. 路由决策：首先将任务分配给 `Clinical_Guideline_Specialist`，查询当前分期下的标准治疗方案。
- 在获取标准指南后，将任务分配给 `Toxicity_Predictor`，评估患者的治疗耐受性和并发症风险。
B. 证据整合：
- 你的核心任务是综合标准指南和患者的个体风险评估结果。
- 最终报告必须给出明确的治疗方案推荐，并附带风险评估结论。
""",
            "Clinical_Guideline_Specialist": {},
            "Toxicity_Predictor": {}
        },
        
        # --- 最终报告修订专家 (Revisor) ---
        "Final_Revisor": {} 
    }
    
    # --- 4. 填充 Level 3 成员角色和工具 ---
    
    # --- 影像团队成员 ---
    prompt["team"]["Imaging_Team"]["MRI_Analyst"] = {
        "tools": [10], # 工具 10: ImagingInterpreterTool
        "prompt": "你是一名 MRI 影像分析专家。请严格根据病历中的 MRI 报告文本，提取 T 分期（肿瘤大小和局部侵犯）和 N 分期（颈部淋巴结）的关键数值和描述。你必须调用 imaging_interpreter 工具来处理报告文本，并将工具的结构化输出作为你的最终回复。"
    }
    prompt["team"]["Imaging_Team"]["PET_Analyst"] = {
        "tools": [10],
        "prompt": "你是一名 PET/CT 分析专家。请专注于评估远处转移。提取 M 分期（远处转移）的关键证据。你必须调用 imaging_interpreter 工具来处理报告文本，并将工具的结构化输出作为你的最终回复。"
    }

    # --- 病理团队成员 ---
    prompt["team"]["Pathology_Team"]["Biopsy_Interpreter"] = {
        "tools": [10], # 复用工具 10
        "prompt": "你是一名活检报告解读专家。请严格根据活检报告，提取 NPC 的组织学类型和肿瘤分级。你必须调用 imaging_interpreter 工具来处理报告文本，并将工具的结构化输出作为你的最终回复。"
    }
    prompt["team"]["Pathology_Team"]["EBV_Marker_Analyst"] = {
        "tools": [11], # 工具 11: RiskCalculatorTool
        "prompt": "你是一名 EB 病毒标记物分析专家。请调用 ebv_risk_calculator 工具，输入 EBV-DNA 拷贝数和当前的 TNM 分期，评估患者的复发风险。你的最终回复必须是工具的结构化输出。"
    }

    # --- 治疗团队成员 ---
    prompt["team"]["Treatment_Team"]["Clinical_Guideline_Specialist"] = {
        "tools": [12], # 工具 12: ClinicalGuidelineTool
        "prompt": "你是一名临床指南专家。根据病理团队提供的最终分期和确诊结果，调用 clinical_guideline_retriever 工具查询标准治疗方案，并返回结构化建议。"
    }
    prompt["team"]["Treatment_Team"]["Toxicity_Predictor"] = {
        "tools": [11], # 复用工具 11
        "prompt": "你是一名治疗毒性评估专家。根据患者的病历数据，调用风险评估工具，预测放化疗的常见并发症风险，并将工具的输出作为你的最终回复。"
    }

    # --- 最终报告修订专家 ---
    # 工具 0 (SerpAPI) 用于可能的额外搜索, 12 用于指南核对
    prompt["team"]["Final_Revisor"] = {
        "tools": [0, 12], 
        "prompt": f"你是最终诊断修订专家。你的任务是接收所有专科团队的报告（位于 intermediate_output 中），整合他们的证据链，对比临床指南，并生成最终的、严格符合 {int_out_format} 格式的 NPC 诊断报告。你必须确保报告的每一个结论都有明确的证据支持。"
    }

    # 最终返回
    prompt["int_out_format"] = int_out_format
    
    return prompt