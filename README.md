### TrieLLM 


### Introduction
A simple example to control LLM for text generations via a Custom Trie (prefix tree).

The large language models (LLMs) generate sentences by selecting tokens from the vocabulary one by one based on their probabilities.

For example, given 3 three sequences to restrict the LLM during generation, allowing it to generate only one (or multiple in the case of Beam Search) from the following three sequences,

        <a_128><b_241><c_146><d_235>
        <a_171><b_57><c_141><d_231>
        <a_135><b_16><c_77><d_23>


the first four choices are fixed and sequential ('<' → 'a' → '_' → '1'), while the fifth choice is restricted to {'2', '7', '3'}.

### Instruction

#### (1) Install requirements 

    >> pip install -r requirements.txt

#### (2) Run generate.py

    >> python generate.py



### ✨✨✨✨ 
- If this repository helps you, please star it. Thank you ~
- If you have any question, please feel free to contact me at kaysenn@163.com.
- 如何通过前缀树来控制LLM做文本生成？[[知乎]](https://zhuanlan.zhihu.com/p/1604769906)

