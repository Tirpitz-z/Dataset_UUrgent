from langchain_core.language_models import LLM
from openai import OpenAI
import httpx
from typing import Dict, List, Optional, Tuple, Union

client = OpenAI(
    base_url="https://api.xiaoai.plus/v1",
    api_key="sk-n2Ycg76HS6faWt1I1e1f5d25C1Fb492e80Bd64Da40C44f59",
    http_client=httpx.Client(
        base_url="https://api.xiaoai.plus/v1",
        follow_redirects=True,
    ),
)


def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [
      {"role": "user", "content": "概括'\u3000\u3000晚上。\n\u3000\u3000宁清提着打包的羊肉汤回到宿舍。\n\u3000\u3000本身她不想打包的，因为知道张酸奶今天出去吃了羊肉汤，而她也不想打包，嫌麻烦，懒得提。但是陈舒听说张酸奶邀请过她一起喝羊肉汤被她拒绝之后，就坚持要她打包一点回去给室友尝尝。\n\u3000\u3000宁清最后还是听了他的。\n\u3000\u3000对于这类事，她总是会听陈舒的。\n\u3000\u3000开门进屋。\n\u3000\u3000走过3号卧室的过道，客厅的灯亮着，张酸奶无力的躺在沙发上，手脚各有一只垂下，搭在地板上，她双目无神的盯着天花板，不知道在想什么。\n\u3000\u3000察觉到动静，她瞬间翻身而起。\n\u3000\u3000静若傻子，动若脱兔。\n\u3000\u3000张酸奶盯着宁清，神情这才恢复：“你回来啦？羊肉汤好喝吗？”\n\u3000\u3000宁清淡淡的看着她，声音很小：\n\u3000\u3000“好喝。”\n\u3000\u3000“我给你打包了一点我今天喝的，我去给你拿。你尝尝哪个更好喝。”张酸奶说着便往饭厅走去，走过宁清身边时察觉到宁清手上也提着有东西，她愣了一下，奇怪的问，“你提的啥子？”\n\u3000\u3000“……”\n\u3000\u3000宁清注视着张酸奶疑惑而纯净的眼睛，仿佛不含一丝杂质，宁清表情依然平静，可眼中却有微光闪烁。\n\u3000\u3000如果自己没听陈舒的，现在会是什么感受呢？\n\u3000\u3000自己会觉得难为情吗？\n\u3000\u3000还是该叫羞愧或者惭愧呢？\n\u3000\u3000“你愣着干啥？”\n\u3000\u3000宁清这才收回目光，抬起手，也把手中的东西拎起来，淡淡说：“我也给你带了一点我今天喝的。”\n\u3000\u3000“哦呀！”\n\u3000\u3000张酸奶很惊奇的看着她：“你今天是不是受什么刺激了？”\n\u3000\u3000“……”\n'"},
      {"role": "assistant", "content": "晚上，宁清听从陈舒的建议，打包了羊肉汤回宿舍给室友张酸奶尝尝。进屋后，她发现张酸奶无精打采地躺在沙发上。张酸奶迅速起身，询问宁清羊肉汤是否好喝，并提到自己也为宁清打包了她喝过的羊肉汤，想要比较哪个更美味。这时，张酸奶注意到宁清手里也提着东西，感到好奇。宁清平静地看着张酸奶，告诉她自己也带了羊肉汤回来。张酸奶对此感到惊讶，调侃宁清是否遇到了什么刺激。"},
      {"role": "user","content": "概括'和煦的阳光照在他面门上，温暖而不灼热，让他想起了之前和同学吃完午饭、躺在操场上虚度午后时光的日子，真是美好。　　果然静心养神。\n　　既然反抗不了……\n　　陈舒连眼睛也闭上了。\n　　隐隐又有脚步声传来，这次停在了他的身边。陈舒努力睁开眼睛，看见一张更青涩的面容，是一个弯腰歪头盯着他的小姑娘，小姑娘站得离他很近，和她姐姐一样，脸上没有多少表情。\n　　“姐夫你在干嘛？”\n　　不远处立马传来宁清的声音：\n　　“不许乱喊。”\n　　小姑娘完全没有理会她，就像是没有听到，依然保持着询问陈舒的动作表情。\n　　陈舒：……\n　　随即见到小姑娘微微抬头，瞄向了旁边的深眠花，沉默了几秒：“姐夫你也闻了深眠花吗？”\n　　又歪头等了一会儿，没等到陈舒的回答，她便自顾自的转身走了。\n　　脚步逐渐远去。\n　　不久后她又回来了。\n　　陈舒只感觉有双手托起了自己的后脑勺，然后两只手变成一只，接着给他脑袋下面垫了一个枕头。\n　　陈舒：……\n　　大约过了半小时。"},
      {"role": "assistant","content": "阳光温柔地洒在陈舒脸上，令他回忆起与同学在操场上的闲暇午后，倍感美好且心灵得到了宁静。尽管处于无法抵抗的状态，他选择闭上眼睛享受这片刻的安宁。随后，一阵脚步声靠近，在他身旁停下，一个小女孩带着稚嫩的脸庞弯腰低头注视着他，与她的姐姐相似，表情不多。小女孩直接称呼他为“姐夫”，询问他在做什么，而宁清则在一旁制止这种称呼。小女孩似乎没听到宁清的话，继续等待陈舒的回答。片刻后，她注意到旁边的深眠花，好奇地问陈舒是否也闻了它，未得到回应便自行离开。但很快，她又回来，细心地帮陈舒的头下垫了个枕头，让他躺得更舒适。约半小时后，故事告一段落。"},

      {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的温度系数，控制输出的随机程度
    )

    return completion



# print(get_completion("概括'他的房间是很整洁的，因为本身就没多少东西，倒是阳台的茶几上散乱的堆着很多书，垃圾桶里有着不少揉成一团的笔记本纸页。墙边摆着张书桌，书桌上的书倒是整齐码放，还有台电脑。　　陈舒拿起一本物理课本，翻看了两下，觉得有些无聊。\n"))


class openai_summarize(LLM):
    model_type: str = "openai"

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "openai"

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = None) -> str:
        res = get_completion(prompt)
        return res

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        _param_dict = {
            "model": self.model_type
        }
        return _param_dict
