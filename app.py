# -*- coding:utf-8 -*-
import streamlit as st
import torch
import pandas as pd
from datetime import datetime

from models.layers import SimpleTARNN
from utils import data_helper as dh

# 设置页面配置
st.set_page_config(
    page_title="题目难度预测系统",
    page_icon="📚",
    layout="wide"
)

# 标题
st.title("📚 题目难度预测系统")
st.markdown("使用 SimpleTARNN 模型预测题目的难度等级")

# 侧边栏
st.sidebar.header("模型配置")

# 初始化历史预测记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 加载模型
@st.cache_resource
def load_model_and_tokenizer(model_path="saved_models/final_model.pt"):
    """加载训练好的模型和分词器"""
    try:
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location='cuda',weights_only=False)
        args = checkpoint['args']
        
        # 加载分词器
        if args.bert_mod == 'local':
            tokenizer = dh.load_bert_tokenizer(local_path=args.bert_path)
        else:
            tokenizer = dh.load_bert_tokenizer(model_name=args.bert_name)
        
        # 初始化模型
        vocab_size = tokenizer.vocab_size
        model = SimpleTARNN(
            args=args,
            vocab_size=vocab_size,
            num_classes=args.num_classes,
            bert_hidden_size=768
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 获取难度映射
        difficulty_map = dh.get_diff_map()
        reverse_diff_map = {v: k for k, v in difficulty_map.items()}
        
        return model, tokenizer, args, reverse_diff_map
        
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None, None, None, None

# 难度等级描述
DIFFICULTY_DESCRIPTION = {
    "容易": "基础题目，适合初学者",
    "较易": "简单题目，需要基本理解",
    "一般": "中等难度题目，需要掌握核心概念",
    "较难": "较难题目，需要综合应用知识",
    "困难": "高难度题目，需要深度思考和复杂推理"
}

def predict_difficulty(model, tokenizer, text, args, max_length=256):
    """预测题目难度"""
    # 编码文本
    encoded = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 获取输入
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    token_type_ids = encoded.get('token_type_ids', torch.zeros_like(input_ids))
    
    # 预测
    with torch.no_grad():
        logits, scores = model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(scores, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, probabilities[0].tolist()

def add_to_history(question_content, prediction, confidence, reverse_diff_map, probabilities):
    """将预测结果添加到历史记录"""
    difficulty_level = reverse_diff_map.get(prediction, "未知")
    
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question_content": question_content[:100] + "..." if len(question_content) > 100 else question_content,
        "full_content": question_content,
        "predicted_difficulty": difficulty_level,
        "confidence": confidence,
        "prediction_value": prediction,
        "probabilities": probabilities
    }
    
    st.session_state['history'].insert(0, history_entry)  # 添加到开头
    
    # 保持历史记录最多100条
    if len(st.session_state['history']) > 100:
        st.session_state['history'] = st.session_state['history'][:100]
    
    return history_entry

def clear_history():
    """清空历史记录"""
    st.session_state['history'] = []

def main():
    # 模型选择
    model_option = st.sidebar.selectbox(
        "选择模型",
        ["默认模型", "自定义模型路径"]
    )
    
    if model_option == "自定义模型路径":
        model_path = st.sidebar.text_input("模型文件路径", "saved_models/final_model.pt")
    else:
        model_path = "saved_models/final_model.pt"
    
    # 加载模型
    if st.sidebar.button("加载模型", type="primary"):
        with st.spinner("正在加载模型..."):
            model, tokenizer, args, reverse_diff_map = load_model_and_tokenizer(model_path)
            
            if model:
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
                st.session_state['args'] = args
                st.session_state['reverse_diff_map'] = reverse_diff_map
                st.sidebar.success("模型加载成功！")
    
    # 检查模型是否已加载
    if 'model' not in st.session_state:
        st.warning("请先加载模型")
        return
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📝 单题预测", "📊 历史预测", "ℹ️ 模型信息"])
    
    # 标签页1: 单题预测
    with tab1:
        st.header("单题难度预测")
        
        col1, col2 = st.columns([3,2])
        
        with col1:
            # 输入区域
            question_content = st.text_area(
                "题目内容",
                height=200,
                placeholder="请输入题目内容...",
                help="输入完整的题目内容进行难度预测"
            )
             
        # 预测按钮
        if st.button("预测难度", type="primary"):
            if not question_content.strip():
                st.error("请输入题目内容！")
            else:
                with st.spinner("正在预测..."):
                    # 进行预测
                    prediction, confidence, probabilities = predict_difficulty(
                        st.session_state['model'],
                        st.session_state['tokenizer'],
                        question_content,
                        st.session_state['args']
                    )
                    
                    # 添加到历史记录
                    history_entry = add_to_history(
                        question_content,
                        prediction,
                        confidence,
                        st.session_state['reverse_diff_map'],
                        probabilities
                    )
                    
                    # 显示结果
                    difficulty_level = history_entry['predicted_difficulty']
                    
                    st.success(f"预测完成！已添加到历史记录")
                    
                    # 结果显示
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            label="📊 预测难度",
                            value=difficulty_level,
                            delta=f"置信度: {confidence:.2%}"
                        )
                        
                        # 显示每个等级的概率
                        st.markdown("### 各等级概率分布")
                        for i in range(5):
                            level_name = st.session_state['reverse_diff_map'].get(i, f"等级{i}")
                            prob = probabilities[i]
                            
                            # 进度条显示概率
                            col_prob1, col_prob2 = st.columns([3, 1])
                            with col_prob1:
                                st.progress(prob, text=f"{level_name}")
                            with col_prob2:
                                st.write(f"{prob:.2%}")
                    
                    with result_col2:
                        # 难度描述
                        st.markdown("### 📖 难度描述")
                        st.info(DIFFICULTY_DESCRIPTION.get(difficulty_level, "未知难度等级"))
                        
                        # 预测详情
                        with st.expander("📋 预测详情"):
                            st.write(f"**预测时间:** {history_entry['timestamp']}")
                            st.write(f"**难度数值:** {prediction}")
                            st.write(f"**置信度:** {confidence:.2%}")
                            st.write(f"**完整概率分布:**")
                            
                            prob_df = pd.DataFrame({
                                '难度等级': [st.session_state['reverse_diff_map'].get(i, f"等级{i}") for i in range(5)],
                                '概率': [f"{p:.2%}" for p in probabilities],
                                '数值': probabilities
                            })
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # 标签页2: 历史预测
    with tab2:
        st.header("历史预测记录")
        
        if not st.session_state['history']:
            st.info("暂无历史预测记录，请在单题预测中进行预测")
        else:
            # 历史记录统计
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            total_count = len(st.session_state['history'])
            most_common = max(
                [(st.session_state['reverse_diff_map'].get(h['prediction_value'], "未知"), 
                  sum(1 for h in st.session_state['history'] if h['prediction_value'] == h['prediction_value'])) 
                 for h in st.session_state['history']], 
                key=lambda x: x[1]
            )
            avg_confidence = sum(h['confidence'] for h in st.session_state['history']) / total_count
            
            with col_stat1:
                st.metric("总预测次数", total_count)
            with col_stat2:
                st.metric("最常见难度", most_common[0])
            with col_stat3:
                st.metric("平均置信度", f"{avg_confidence:.2%}")
            
            # 清空历史按钮
            if st.button("🗑️ 清空历史记录", type="secondary"):
                clear_history()
                st.rerun()
            
            # 搜索和筛选功能
            st.subheader("📋 历史记录列表")
            
            search_col1, search_col2 = st.columns([2, 1])
            with search_col1:
                search_text = st.text_input("搜索题目内容", placeholder="输入关键词搜索...")
            with search_col2:
                difficulty_filter = st.selectbox(
                    "筛选难度等级",
                    ["全部"] + [st.session_state['reverse_diff_map'].get(i, f"等级{i}") for i in range(5)]
                )
            
            # 显示历史记录表格
            filtered_history = st.session_state['history']
            
            if search_text:
                filtered_history = [h for h in filtered_history if search_text.lower() in h['question_content'].lower()]
            
            if difficulty_filter != "全部":
                filtered_history = [h for h in filtered_history if h['predicted_difficulty'] == difficulty_filter]
            
            if filtered_history:
                # 创建显示用的DataFrame
                display_data = []
                for i, entry in enumerate(filtered_history):
                    display_data.append({
                        "序号": i + 1,
                        "预测时间": entry['timestamp'],
                        "题目内容": entry['question_content'],
                        "预测难度": entry['predicted_difficulty'],
                        "置信度": f"{entry['confidence']:.2%}",
                        "难度数值": entry['prediction_value'],
                        "完整内容": entry['full_content']
                    })
                
                df = pd.DataFrame(display_data)
                
                # 分页显示
                page_size = 10
                total_pages = (len(df) + page_size - 1) // page_size
                
                page_num = st.number_input(
                    f"页码 (共{total_pages}页)", 
                    min_value=1, 
                    max_value=total_pages if total_pages > 0 else 1,
                    value=1
                )
                
                start_idx = (page_num - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                
                # 显示当前页数据
                st.dataframe(
                    df.iloc[start_idx:end_idx][["序号", "预测时间", "题目内容", "预测难度", "置信度"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                # 查看详细内容
                selected_idx = st.selectbox(
                    "选择记录查看详情",
                    options=[f"{i+1}. {row['题目内容']}" for i, row in df.iterrows()],
                    index=0
                )
                
                if selected_idx:
                    selected_num = int(selected_idx.split(".")[0]) - 1
                    selected_entry = filtered_history[selected_num]
                    
                    with st.expander("📄 查看详情", expanded=True):
                        st.write("**完整题目内容:**")
                        st.text_area("", selected_entry['full_content'], height=150, disabled=True)
                        
                        st.write("**预测结果:**")
                        col_detail1, col_detail2 = st.columns(2)
                        with col_detail1:
                            st.write(f"预测难度: **{selected_entry['predicted_difficulty']}**")
                            st.write(f"置信度: **{selected_entry['confidence']:.2%}**")
                        with col_detail2:
                            st.write(f"预测时间: {selected_entry['timestamp']}")
                            st.write(f"难度数值: {selected_entry['prediction_value']}")
                        
                        # 概率分布图
                        st.write("**概率分布:**")
                        prob_data = pd.DataFrame({
                            '难度等级': [st.session_state['reverse_diff_map'].get(i, f"等级{i}") for i in range(5)],
                            '概率': selected_entry['probabilities']
                        })
                        st.bar_chart(prob_data.set_index('难度等级'))
                
                # 导出历史记录
                st.subheader("📤 导出历史记录")
                export_col1, export_col2 = st.columns([2, 1])
                
                with export_col1:
                    export_format = st.radio("导出格式", ["CSV", "JSON"])
                
                with export_col2:
                    if st.button("导出数据"):
                        export_df = pd.DataFrame([{
                            "预测时间": h['timestamp'],
                            "题目内容": h['full_content'],
                            "预测难度": h['predicted_difficulty'],
                            "难度数值": h['prediction_value'],
                            "置信度": h['confidence'],
                            "容易概率": h['probabilities'][0],
                            "较易概率": h['probabilities'][1],
                            "一般概率": h['probabilities'][2],
                            "较难概率": h['probabilities'][3],
                            "困难概率": h['probabilities'][4]
                        } for h in filtered_history])
                        
                        if export_format == "CSV":
                            csv = export_df.to_csv(index=False)
                            st.download_button(
                                label="下载CSV文件",
                                data=csv,
                                file_name=f"难度预测历史_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            json_data = export_df.to_json(orient='records', force_ascii=False)
                            st.download_button(
                                label="下载JSON文件",
                                data=json_data,
                                file_name=f"难度预测历史_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            else:
                st.info("没有找到匹配的历史记录")
    
    # 标签页3: 模型信息
    with tab3:
        st.header("模型信息")
        
        if 'args' in st.session_state:
            # 模型配置信息
            st.subheader("📊 模型配置")
            
            config_data = {
                "模型名称": "SimpleTARNN",
                "BERT模型": st.session_state['args'].bert_mod,
                "BERT路径/名称": st.session_state['args'].bert_path if st.session_state['args'].bert_mod == 'local' else st.session_state['args'].bert_name,
                "RNN层数": st.session_state['args'].rnn_layers,
                "RNN维度": st.session_state['args'].rnn_dim,
                "注意力类型": st.session_state['args'].attention_type,
                "分类类别数": st.session_state['args'].num_classes,
                "学习率": st.session_state['args'].learning_rate,
                "批次大小": st.session_state['args'].batch_size,
                "Dropout率": st.session_state['args'].dropout_rate
            }
            
            for key, value in config_data.items():
                st.info(f"**{key}:** {value}")
            
            # 难度映射
            st.subheader("🎯 难度等级映射")
            difficulty_map = dh.get_diff_map()
            
            for level, value in difficulty_map.items():
                st.write(f"- **{level}** → 数值标签: {value}")
        
        # 使用说明
        st.subheader("📖 使用说明")
        st.markdown("""
        1. **加载模型**: 在侧边栏点击"加载模型"按钮
        2. **单题预测**: 在"单题预测"标签页输入题目内容进行预测
        3. **历史预测**: 在"历史预测"标签页查看和管理所有预测记录
        4. **历史记录功能**:
           - 自动保存每次预测结果
           - 支持搜索和筛选
           - 可以导出为CSV或JSON格式
           - 最多保存100条记录
        """)
        
        # 注意事项
        st.subheader("⚠️ 注意事项")
        st.warning("""
        - 确保BERT模型路径正确
        - 预测结果仅供参考，实际难度需结合专家判断
        - 模型在训练数据分布之外的题目上可能表现不佳
        - 历史记录仅在当前会话中有效，刷新页面会清空历史
        """)

if __name__ == "__main__":
    main()