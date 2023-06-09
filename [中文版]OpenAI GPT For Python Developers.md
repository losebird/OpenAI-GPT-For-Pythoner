### GPT 如何运作？
Generative Pre-trained Transformer 或 GPT 是一种生成文本模型。该模型能够根据收到的输入预测接下来会发生什么，从而生成新文本。

GPT-3 是另一种模型，它显然比之前的任何其他 GPT 模型（包括 GPT-1 和 GPT-2）更大、性能更高。

第 3 代在大量文本语料库上进行训练，例如书籍、文章以及 Reddit 和其他论坛等可公开访问的网站，它使用这些训练数据来学习单词和短语之间的模式和关系。

GPT-3 的关键创新在于其惊人的规模——拥有惊人的 1750 亿个参数，使其成为有史以来最庞大、最强大的语言模型之一。它在如此庞大的数据集上进行的广泛训练使其能够生成类似人类的文本，执行各种自然语言处理任务，并以令人印象深刻的准确性完成任务。

GPT 是一种称为 Transformer 的神经网络，专为自然语言处理任务而设计。 Transformer 的架构基于一系列自注意力机制，允许模型并行处理输入文本，并根据上下文权衡每个单词或标记的重要性。

自注意力是自然语言处理 (NLP) 深度学习模型中使用的一种机制，它允许模型在进行预测时权衡一个句子或多个句子的不同部分的重要性。作为 Transformer 架构的一部分，它使神经网络在执行 NLP 任务时能够达到令人满意的性能。

这是使用 Hugging Face 转换器进行 GPT-2 推理的示例。
```python
from transformers import pipeline

generator = pipeline('text-generation', model = 'gpt2')
generator("Hello, I'm a language model", max_length = 30, num_return_sequences=3)
## [{'generated_text': "Hello, I'm a language modeler. So while writing this, when I went out to meet my wife or come home she told me that my"},
## {'generated_text': "Hello, I'm a language modeler. I write and maintain software in Python. I love to code, and that includes coding things that require writing"}, {...}]
...
```
默认情况下，模型没有记忆，这意味着每个输入都是独立处理的，不会从以前的输入中继承任何信息。当 GPT 生成文本时，它不会根据先前的输入对接下来应该出现的内容有任何先入为主的观念。相反，它根据每个单词是给定先前输入的下一个可能单词的概率来生成每个单词。这会产生令人惊讶且富有创意的文本。

这是使用 GPT 模型根据用户输入生成文本的另一个代码示例。

GPT 如何运作？
```python
# 导入必要的库
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练的GPT-2 分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 将模型设置为评估模式
model.eval()

# 定义需要让模型完成的提示语
prompt = input("You: ")

# 标记提示并生成文本
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=50, do_sample=True)

# 解码并生成文本输出在console界面
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("AI: " + generated_tex)
```
GPT-3 被设计为通用语言模型，这意味着它可以用于各种自然语言处理任务，例如语言翻译、文本摘要和问答。 OpenAI 已经过 GPT-3 训练，但您可以根据微调的数据集建立自己的训练模型。除了模型可以协助的默认任务（例如生成文本、诗歌和故事）之外，这还可以实现更具创造性和特定的用例。它还可以用于构建特定领域专家的聊天机器人、其他对话界面等等！

在本指南中，我们将深入探讨 OpenAI GPT-3，包括如何有效地利用公司和 AI/ML 社区提供的 API 和工具来制作功能强大且富有创意的工具和系统的原型。这不仅会提高您使用 GPT-3 API 的熟练程度，还会拓宽您对自然语言处理和其他相关领域中使用的概念和技术的理解。

GPT-3 因其广泛的规模而被认为是自然语言处理领域的重大进步。然而，一些专家对该模型可能会产生带有偏见或破坏性的内容表示担忧。与任何技术一样，退后一步并考虑其使用的道德影响是非常重要的。本指南将不涉及伦理问题，而只关注实际方面。

## 准备开发环境

### 准备开发环境安装 Python、pip 和虚拟开发环境

显然，你需要有 Python。在本书中，我们将使用 Python 3.9.5。

我们还将使用 pip，Python 的包安装程序。您可以使用 pip 从 Python 包索引和其他索引安装包。

您不需要在系统中安装 Python 3.9.5，因为我们将创建一个虚拟开发环境。因此，无论您拥有什么版本，您的开发环境都将与您的系统隔离，并将使用相同的 Python 3.9.5。

如果未安装 Python，请转至 www.python.org/downloads/¹¹，下载并安装其中一个 Python 3.x 版本。根据您的操作系统，您将必须遵循不同的说明。

为了管理我们的开发环境，我们将使用 virtualenvwrapper¹²。您可以在官方文档中找到安装说明¹³。

获取它的最简单方法是使用 pip：
```python
pip install virtualenvwrapper
```
注意：如果你习惯了 virtualenv、Poetry 或任何其他包管理器，你可以继续使用它。在这种情况下无需安装 virtualenvwrapper。

如果没有安装 pip，最简单的安装方式是使用官方文档中提供的脚本：
从 https://bootstrap.pypa.io/get-pip.py 下载脚本

打开终端/命令提示符，cd 到包含 get-pip.py 文件的文件夹并运行:
```shell
# macOS and Linux users
python get-pip.py

# windows users
py get-pip.py
```
总之，您应该在系统中安装这些软件包：
• Python 

• pip

• virtualenvwrapper（virtualenv 或您喜欢的任何其他包管理器）

我强烈建议 Windows 用户创建一个内置 Linux 的虚拟机，因为本书中提供的大多数示例都是在 Linux Mint 系统上运行和测试的。

接下来，让我们创建一个虚拟环境：
```shell
mkvirtualenv -p python3.9 chatgptforpythondevelopers

 创建后，激活虚拟环境：

workon chatgptforpythondevelopers
```
### 获取您的 OpenAI API 密钥

下一步是创建 API 密钥，使您能够访问 OpenAI 提供的官方 API。

转到 https://openai.com/api 并创建一个帐户。

按照提供的说明创建一个帐户，然后在此处创建您的 API 密钥：

API 密钥应该属于一个组织，系统会要求您创建一个组织。在本书中，我们将其称为：“LearningGPT”。

将生成的密钥保存在安全且可访问的地方。您将无法通过您的 OpenAI 帐户再次查看它。

### 安装官方 Python 绑定

您可以使用任何语言的 HTTP 请求与 API 交互，无论是通过官方 Python 绑定、我们的官方 Node.js 库还是社区维护的库。

在本书中，我们将使用 OpenAI 提供的官方库。另一种选择是使用 Chronology，这是一个由 OthersideAI 提供的非官方库。不过这个库貌似不再更新了。

要安装官方 Python 绑定，请运行以下命令：
```shell
pip install openai
 
  确保您在我们之前创建的虚拟环境中安装库。
```

### 测试我们的 API 密钥

为了验证一切正常，我们将执行 curl 调用。

让我们先将 API 密钥和组织 ID 存储在 .env 文件中：
```shell
1 cat << EOF > .env
2 API_KEY=xxx
3 ORG_ID=xxx
4 EOF
    在执行上述命令之前，请确保通过各自的值更新 API_KEY 和 ORG_ID 变量。
现在您可以执行以下命令：
1 source .env

2 curl https://api.openai.com/v1/models \
3 -H 'Authorization: Bearer '$API_KEY'' \
4 -H 'OpenAI-Organization: '$ORG_I
```
您可以跳过获取变量并直接在 curl 命令中使用它们：
```shell
1 curl https://api.openai.com/v1/models \
2 -H 'Authorization: Bearer xxxx' \
3 -H 'OpenAI-Organization: xxxx'
    如果您的 OpenAI 帐户中有一个组织，则可以在不提及组织 ID 的情况下执行相同的命令。
1 curl https://api.openai.com/v1/models -H 'Authorization: Bearer
curl 命令应该会为您提供 API 提供的模型列表，例如“davinci”、“ada”等。
```
要使用 Python 代码测试 API，我们可以执行以下代码：
```py
import os
import openai

# 从 .evn 文件中读取变量，获得API_KEY 和 ORG_ID
with open(".env") as env:
    for line in env:
        key, value = line.strip().split("=")
        os.environ[key] = value

# 初始化API KEY 和 ORG ID
openai.api_key = os.environ.get("API_KEY")
openai.organization = os.environ.get("ORG_ID")

# 调用API并列出模型
models = openai.Model.list()
print(models)
```
我们将来会使用几乎相同的方法，所以让我们创建一个可重用的函数来初始化 API。我们可以这样写我们的代码：
```py
import os
import openai

def init_api():
    with open('.evn') as env:
        for line in env:
            key, value = line.strip().split('=')
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

models = openai.Model.list()
print(models)
```

## 可用型号

### 三个主要模型

如果我们可以这样称呼它们，则有主要模型或模型系列：

• GPT-3 

• Codex 

• 内容过滤器模型

您可能会在某些在线文档资源中找到其他模型的名称，这可能会造成混淆。其中许多文档都是关于旧版本的，而不是 GPT-3。另请注意，除了三个主要模型外，您还可以创建和微调自己的模型。

GPT-3：处理和生成自然语言GPT-3 模型能够理解人类语言和看似自然语言的文本。该模型系列包含 4 个模型（A、B、C、D），它们或多或少速度快且性能高。

• D：text-davinci-003 

• C：text-curie-001 

• B：text-babbage-001 

• A：text-ada-001

如前所述，每个都有不同的功能、定价和准确性。

OpenAI 建议尝试使用达芬奇模型，然后尝试其他能够以低得多的成本执行大量类似任务的模型。

#### text-​​davinci-003

这是功能最强大的 GPT-3 模型，因为它可以执行所有其他模型可以执行的操作。此外，与其他产品相比，它提供了更高的质量。这是最新的模型，因为它是使用截至 2021 年 6 月的数据进行训练的。

它的优势之一是允许请求多达 4k 个tokens。它还支持在文本中插入补全。

我们将在本指南后面更详细地定义token。现在，只知道它们决定了用户请求的长度。

#### text-curie-001
text-curie-001 模型是第二个最强大的 GPT-3 模型，因为它支持多达 2048 个tokens。

它的优点是比 text-davinci-003 更具成本效益，但仍然具有很高的准确性。

它使用截至 2019 年 10 月的数据进行训练，因此它的准确性略低于 text-davinci-003。

它可能是翻译、复杂分类、文本分析和摘要的不错选择。

#### text-babbage-001
与 Curie 相同：截至 2019 年 10 月的 2,048 个tokens和数据训练。

该模型对于更简单的分类和语义分类是有效的。

#### text-ada-001
与 Curie 相同：截至 2019 年 10 月的 2,048 个tokens和数据训练。

该模型非常快速且具有成本效益，是最简单的分类、文本提取和地址更正的首选。

### Codex: 理解和生成计算机代码

OpenAI 提出了两种用于理解和生成计算机代码的 Codex 模型：code-davinci002 和 code-cushman-001。

Codex 是为 GitHub Copilot 提供支持的模型。精通Python、JavaScript、Go、Perl、PHP、Ruby、Swift、TypeScript、SQL、Shell等十几种编程语言。

Codex 能够理解以自然语言表达的基本指令，并代表用户执行请求的任务。

Codex 有两种型号：

• code-davinci-002 

• code-cushman-001

#### code-davinci-002
Codex 模型是最有能力的。它擅长将自然语言翻译成代码。不仅完善了代码，还支持补充元素的插入。它最多可以处理 8,000 个令牌，并接受了截至 2021 年 6 月的数据训练。

#### code-cushman-001
Cushman 强大而快速。即使 Davinci 在分析复杂任务方面更强大，这个模型也有很多代码生成任务的能力。

它也比 Davinci 更快、更实惠。

### Content Filter

顾名思义，这是一个针对敏感内容的过滤器。

使用此过滤器，您可以检测 API 生成的敏感或不安全文本。此过滤器可以将文本分为 3 类：

• 安全的，

• 敏感， 

• 不安全。

如果您正在构建一个将由您的用户使用的应用程序，您可以使用过滤器来检测模型是否返回任何不适当的内容。

### 列出所有模型
使用 API 模型端点，您可以列出所有可用模型。

让我们看看它在实践中是如何工作的：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

models = openai.Model.list()
print(models)
```
如您所见，我们仅使用包 openai 的模块模型中的函数 list()。结果，我们得到类似于以下的列表：
```json
{
"data": [
    {
    "created": 1649358449,
    "id": "babbage",
    "object": "model",
    "owned_by": "openai",
    "parent": null,
    "permission": [
            {
            "allow_create_engine": false,
            "allow_fine_tuning": false,
            "allow_logprobs": true,
            "allow_sampling": true,
            "allow_search_indices": false,
            "allow_view": true,
            "created": 1669085501,
            "group": null,
            "id": "modelperm-49FUp5v084tBB49tC4z8LPH5",
            "is_blocking": false,
            "object": "model_permission",
            "organization": "*"
            }
        ],
        "root": "babbage"
    },
[...]
[...]
[...]
[...]
[...]
    {
    "created": 1642018370,
    "id": "text-babbage:001",
    "object": "model",
    "owned_by": "openai",
    "parent": null,
    "permission": [
            {
                "allow_create_engine": false,
                "allow_fine_tuning": false,
                "allow_logprobs": true,
                "allow_sampling": true,
                "allow_search_indices": false,
                "allow_view": true,
                "created": 1642018480,
                "group": null,
                "id": "snapperm-7oP3WFr9x7qf5xb3eZrVABAH",
                "is_blocking": false,
                "object": "model_permission",
                "organization": "*"
            }
        ],
        "root": "text-babbage:001"
    }
 ],
 "object": "list"
 }
```
让我们只打印模型的 ID：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

models = openai.Model.list()
for model in models['data']:
    print(model['id'])
```
结果应该是：
```json
1 babbage
2 ada
3 davinci
4 text-embedding-ada-002
5 babbage-code-search-code
6 text-similarity-babbage-001
7 text-davinci-003
8 text-davinci-001
9 curie-instruct-beta
10 babbage-code-search-text
11 babbage-similarity
12 curie-search-query
13 code-search-babbage-text-001
14 code-cushman-001
15 code-search-babbage-code-001
16 audio-transcribe-deprecated
17 code-davinci-002
18 text-ada-001
19 text-similarity-ada-001
20 text-davinci-insert-002
21 ada-code-search-code
22 text-davinci-002
23 ada-similarity
24 code-search-ada-text-001
25 text-search-ada-query-001
26 text-curie-001
27 text-davinci-edit-001
28 davinci-search-document
29 ada-code-search-text
30 text-search-ada-doc-001
31 code-davinci-edit-001
32 davinci-instruct-beta
33 text-similarity-curie-001
34 code-search-ada-code-001
35 ada-search-query
36 text-search-davinci-query-001
37 davinci-search-query
38 text-davinci-insert-001
39 babbage-search-document
40 ada-search-document
41 text-search-babbage-doc-001
42 text-search-curie-doc-001
43 text-search-curie-query-001
...
```
我们已经知道 Ada、Babbage、Curie、Cushman 和 Davinci。

我们还看到了 Codex。每个在其 ID 中包含代码的模型都是 Codex 的一部分。

### 使用哪种模型？
达芬奇模型是迄今为止最好的模型，但它们也是最昂贵的。因此，如果优化成本不是您的首要任务，并且您希望关注质量，那么 Davinci 是您的最佳选择。更具体地说，text-davinci-003 是最强大的模型。

与 davinci 相比，text-davinci-003 是一种更新、功能更强大的模型，专为指令跟踪任务和零样本场景而设计，我们将在本指南后面探讨这一概念。

但是，请记住，对于某些特定用例，Davinci 模型并不总是首选。我们将在本指南的后面部分看到更多详细信息。

为了优化成本，其他模型（例如 Curie）是不错的选择，尤其是在您执行文本摘要或数据提取等简单请求时。

这适用于 GPT-3 和 Codex。

内容过滤器是可选的，但如果您正在构建公共应用程序，则强烈建议使用它。您不希望您的用户收到不适当的结果。

### 下一步是什么？
总结本章，像 text-davinci-003 这样的 Davinci 模型是最好的模型。对于大多数用例，OpenAI 推荐使用 text-davinci-003。其他型号如Curie 在某些用例中表现非常出色，成本仅为 Davinci 的 1/10 左右。

定价模型清晰易懂。它可以在官方网站上找到。此外，当您注册 OpenAI 时，您将获得 18 美元的免费信用额度，可在前 3 个月内使用。

在本指南中，我们将遵循 OpenAI 的建议并使用新模型，而不是已弃用的模型。

随着我们深入研究这些模型，您将更好地了解如何使用 API 创建、生成和编辑文本和图像，无论是使用基础模型还是您自己的微调模型。当我们做更多实际示例时，您将对代币和定价有更好的理解，到最后，您将能够使用此 AI 创建可用于生产的智能应用程序。

## 使用 GPT Text Completions
一旦您验证了您的应用程序，您就可以开始使用 OpenAI API 来执行完成。为此，您需要使用 OpenAI Completion API。

OpenAI Completion API 使开发人员能够访问 OpenAI 的数据集和模型，从而毫不费力地完成任务。

首先提供句子的开头。然后该模型将预测一个或多个可能的完成，每个都有一个相关的分数。

### 一个基础完成示例
例如，我们向 API 提供句子“Once upon a time”，它会返回下一个可能的句子。

激活开发环境：
```shell
workon chatgptforpythondevelo
```
创建一个新的 Python 文件“app.py”，您将在其中添加以下代码：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="txt-davinci-003",
    prompt="Once upon a time",
    max_tokens=7,
    temperature=0
)

print(next)
```
执行文件：
```shell
python app.py
```
API 应返回如下内容：
```json
1 {
2 "choices": [
3 {
4 "finish_reason": "length",
5 "index": 0,
6 "logprobs": null,
7 "text": " there was a little girl named Alice"
8 }
9 ],
10 "created": 1674510189,
11 "id": "cmpl-6bysnKOUj0QWOu5DSiJAGcwIVMfNh",
12 "model": "text-davinci-003",
13 "object": "text_completion",
14 "usage": {
15 "completion_tokens": 7,
16 "prompt_tokens": 4,
17 "total_tokens": 11
18 }
19 }
```
在上面的例子中，我们只有一个 choices ：“有一个叫爱丽丝的小女孩”。此结果的索引为 0。

API 还返回了“finish_reason”，在本例中为“length”。

输出的长度由 API 根据用户提供的“max_tokens”值确定。在我们的例子中，我们将此值设置为 7。

注意：根据定义，标记是输出文本中的常见字符序列。一个好的记忆方法是，对于普通英语单词，一个标记通常表示大约 4 个文本字母。这意味着 100 个标记与 75 个单词大致相同。掌握这一点将有助于您理解定价。在本书的后面，我们将更深入地研究定价细节。

### 控制输出的令牌计数
让我们用一个更长的例子来测试，这意味着更多的标记（15）：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=15,
    temperature=0
)

print(next)
```
API返回了更长的文本：
```json
1 {
2 "choices": [
3 {
4 "finish_reason": "length",
5 "index": 0,
6 "logprobs": null,
7 "text": " there was a little girl named Alice. She lived in a small village wi\
8 th"
9 }
10 ],
11 "created": 1674510438,
12 "id": "cmpl-6bywotGlkujbrEF0emJ8iJmJBEEnO",
13 "model": "text-davinci-003",
14 "object": "text_completion",
15 "usage": {
16 "completion_tokens": 15,
17 "prompt_tokens": 4,
18 "total_tokens": 19
19 }
20 }
```
### Logprobs
为了增加可能性，我们可以使用“logprobs”参数。例如，将 logprobs 设置为 2 将返回每个标记的两个版本。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=15,
    temperature=0,
    logprobs=3,
)

print(next)
```
这是 API 返回的内容：
```json
1 {
2 "choices": [
3 {
4 "finish_reason": "length",
5 "index": 0,
6 "logprobs": {
7 "text_offset": [
8 16,
9 22,
10 26,
11 28,
12 35,
13 40,
14 46,
15 52,
16 53,
17 57,
18 63,
19 66,
20 68,
21 74,
22 82
23 ],
24 "token_logprobs": [
25 -0.9263134,
26 -0.2422086,
27 -0.039050072,
28 -1.8855333,
29 -0.15475112,
30 -0.30592665,
31 -1.9697434,
32 -0.34726024,
33 -0.46498245,
34 -0.46052673,
35 -0.14448218,
36 -0.0038384167,
37 -0.029725535,
38 -0.34297562,
39 -0.4261593
40 ],
41 "tokens": [
42 " there",
43 " was",
44 " a",
45 " little",
46 " girl",
47 " named",
48 " Alice",
49 ".",
50 " She",
51 " lived",
52 " in",
53 " a",
54 " small",
55 " village",
56 " with"
57 ],
58 "top_logprobs": [
59 {
60 "\n": -1.1709108,
61 " there": -0.9263134
62 },
63 {
64 " lived": -2.040701,
65 " was": -0.2422086
66 },
67 {
68 " a": -0.039050072,
69 " an": -3.403554
70 },
71 {
72 " little": -1.8855333,
73 " young": -2.02082
74 },
75 {
76 " boy": -2.449015,
77 " girl": -0.15475112
78 },
79 {
80 " named": -0.30592665,
81 " who": -1.7700866
82 },
83 {
84 " Alice": -1.9697434,
85 " Sarah": -2.9232333
86 },
87 {
88 " who": -1.3002346,
89 ".": -0.34726024
90 },
91 {
92 " Alice": -1.2721952,
93 " She": -0.46498245
94 },
95 {
96 " lived": -0.46052673,
97 " was": -1.7077477
98 },
99 {
100 " in": -0.14448218,
101 " with": -2.0538774
102 },
103 {
104 " a": -0.0038384167,
105 " the": -5.8157005
106 },
107 {
108 " quaint": -4.941383,
109 " small": -0.029725535
110 },
111 {
112 " town": -1.454277,
113 " village": -0.34297562
114 },
115 {
116 " in": -1.7855972,
117 " with": -0.4261593
118 }
119 ]
120 },
121 "text": " there was a little girl named Alice. She lived in a small village wi\
122 th"
123 }
124 ],
125 "created": 1674511658,
126 "id": "cmpl-6bzGUTuc5AmbjsoNLNJULTaG0WWUP",
127 "model": "text-davinci-003",
128 "object": "text_completion",
129 "usage": {
130 "completion_tokens": 15,
131 "prompt_tokens": 4,
132 "total_tokens": 19
133 }
134 }
```
您可以看到每个标记都有一个与之关联的概率或分数。 API 将在“\n”和“there”之间返回“there”，因为 -1.1709108 小于 -0.9263134。

API 将选择“was”而不是“lived”，因为 -0.2422086 大于 -2.040701。同样，其他值也是如此。
```json
1 {
2 "\n": -1.1709108,
3 " there": -0.9263134
4 },
5 {
6 " lived": -2.040701,
7 " was": -0.2422086
8 },
9 {
10 " a": -0.039050072,
11 " an": -3.403554
12 },
13 {
14 " little": -1.8855333,
15 " young": -2.02082
16 },
17 {
18 " boy": -2.449015,
19 " girl": -0.15475112
20 },
21 {
22 " named": -0.30592665,
23 " who": -1.7700866
24 },
25 {
26 " Alice": -1.9697434,
27 " Sarah": -2.9232333
28 },
29 {
30 " who": -1.3002346,
31 ".": -0.34726024
32 },
33 {
34 " Alice": -1.2721952,
35 " She": -0.46498245
36 },
37 {
38 " lived": -0.46052673,
39 " was": -1.7077477
40 },
41 {
42 " in": -0.14448218,
43 " with": -2.0538774
44 },
45 {
46 " a": -0.0038384167,
47 " the": -5.8157005
48 },
49 {
50 " quaint": -4.941383,
51 " small": -0.029725535
52 },
53 {
54 " town": -1.454277,
55 " village": -0.34297562
56 },
57 {
58 " in": -1.7855972,
59 " with": -0.4261593
60 }
```
每个标记都有两个可能的值。 API返回每一个的概率以及由概率最高的token组成的句子。
```json
1 "tokens": [
2 " there",
3 " was",
4 " a",
5 " little",
6 " girl",
7 " named",
8 " Alice",
9 ".",
10 " She",
11 " lived",
12 " in",
13 " a",
14 " small",
15 " village",
16 " with"
17 ],
```
我们可以将大小增加到 5。根据 OpenAI，logprobs 的最大值为 5。如果您需要更多，请联系他们的帮助中心并说明您的用例。

### 控制创造力：The Sampling Temperature
我们可以自定义的下一个参数是温度。这可以用来使模型更具创意，但创意会带来一些风险。

对于更有创意的应用，我们可以使用更高的温度，例如 0.2、0.3、0.4、0.5 和 0.6。最高温度为2。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=15,
    temperature=2,
)

print(next)
```
API返回：
```json
1 {
2 "choices": [
3 {
4 "finish_reason": "length",
5 "index": 0,
6 "logprobs": null,
7 "text": " there lived\ntwo travellers who ravret for miles through desert fore\
8 st carwhen"
9 }
10 ],
11 "created": 1674512348,
12 "id": "cmpl-6bzRc4nJXBKBaOE6pr5d4bLCLI7N5",
13 "model": "text-davinci-003",
14 "object": "text_completion",
15 "usage": {
16 "completion_tokens": 15,
17 "prompt_tokens": 4,
18 "total_tokens": 19
19 }
20 }
```
温度设置为其最大值，因此执行相同的脚本应该返回不同的结果。这就是创造力发挥作用的地方。

### 使用“top_p”采样
或者，我们可以使用 top_p 参数。例如，使用 0.5 意味着只考虑概率质量最高的标记，占 50%。使用 0.1 意味着考虑具有最高概率质量的标记，包括 10%。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=15,
    top_p=0.9,
)

print(next)
```
建议使用 top_p 参数或温度参数，但不要同时使用两者。

top_p参数也称为nucleus sampling或top-p sampling。

### 流式传输结果
我们可以在 OpenAI 中使用的另一个常见参数是流。可以指示 API 返回令牌流而不是包含所有令牌的块。在这种情况下，API 将返回一个生成器，该生成器按照生成令牌的顺序生成令牌。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=7,
    stream=True,
)

# 将返回 <class 'generator'>
print(type(next))

# * 将解压生成器
print(*next, sep='\n')
```
这应该打印出下一个类型，即<class 'generator'>，然后是紧随其后的标记。
```json
1 {
2 "choices": [
3 {
4 "finish_reason": null,
5 "index": 0,
6 "logprobs": null,
7 "text": " there"
8 }
9 ],
10 "created": 1674594500,
11 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
12 "model": "text-davinci-003",
13 "object": "text_completion"
14 }
15 {
16 "choices": [
17 {
18 "finish_reason": null,
19 "index": 0,
20 "logprobs": null,
21 "text": " was"
22 }
23 ],
24 "created": 1674594500,
25 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
26 "model": "text-davinci-003",
27 "object": "text_completion"
28 }
29 {
30 "choices": [
31 {
32 "finish_reason": null,
33 "index": 0,
34 "logprobs": null,
35 "text": " a"
36 }
37 ],
38 "created": 1674594500,
39 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
40 "model": "text-davinci-003",
41 "object": "text_completion"
42 }
43 {
44 "choices": [
45 {
46 "finish_reason": null,
47 "index": 0,
48 "logprobs": null,
49 "text": " girl"
50 }
51 ],
52 "created": 1674594500,
53 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
54 "model": "text-davinci-003",
55 "object": "text_completion"
56 }
57 {
58 "choices": [
59 {
60 "finish_reason": null,
61 "index": 0,
62 "logprobs": null,
63 "text": " who"
64 }
65 ],
66 "created": 1674594500,
67 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
68 "model": "text-davinci-003",
69 "object": "text_completion"
70 }
71 {
72 "choices": [
73 {
74 "finish_reason": null,
75 "index": 0,
76 "logprobs": null,
77 "text": " was"
78 }
79 ],
80 "created": 1674594500,
81 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
82 "model": "text-davinci-003",
83 "object": "text_completion"
84 }
85 {
86 "choices": [
87 {
88 "finish_reason": "length",
89 "index": 0,
90 "logprobs": null,
91 "text": " very"
92 }
93 ],
94 "created": 1674594500,
95 "id": "cmpl-6cKoeTcDzK6NnemYNgKcmpvCvT9od",
96 "model": "text-davinci-003",
97 "object": "text_completion"
98 }
```
如果只想获取文本，可以使用类似下面的代码：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=7,
    stream=True,
)

# 一个一个读取生成器文本元素
for i in next:
    print(i['choices'][0]['text']) 
```
应该打印如下内容：
```text
1 there
2 was
3 a
4 small
5 village
6 that
7 was
```

### 控制重复性：频率和存在惩罚
Completions API 具有两个功能，可用于阻止过于频繁地建议相同的词。这些功能通过向 logits（显示建议单词的可能性的数字）添加奖励或惩罚来改变某些单词被建议的机会。

可以使用两个参数启用这些功能：

    • presence_penalty 是一个介于-2.0 和2.0 之间的数字。如果这个数字是正数，它会让模型更有可能谈论新话题，因为如果它使用已经使用过的词，它会受到惩罚。

    • Frequency_penalty 是介于-2.0 和2.0 之间的数字。正值使模型不太可能重复已使用的同一行文本。

为了理解这些参数的作用，让我们在下面的代码中使用它们：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=100,
    frequency_penalty=2.0,
    presence_penalty=2.0
)

print("=== Frequency and presence penalty 2.0 ===")
print(next["choices"][0]["text"])

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=100,
    frequency_penalty=-2.0,
    presence_penalty=-2.0
)

print("=== Frequency and presence penalty -2.0 ===")
print(next["choices"][0]["text"])
```
正如你所看到的，第一次执行会在文本中产生更多的多样性（frequency_penalty=2.0 and presence_penalty=2.0），而第二次则完全相反（frequency_penalty=-2.0,presence_penalty=-2.0）。

执行上述代码后，输出如下：
```text
1 === Frequency and presence penalty 2.0 ===
2
3 there was a beautiful princess named Cinderella.
4 She lived with her wicked stepmother and two jealous stepsisters who treated her lik\
5 e their servant. One day, an invitation arrived to the palace ball where all of the \
6 eligible young ladies in town were invited by the prince himself! Cinderella's deter\
7 mination kept fueling when she noticed how cruelly she been mistreated as if it gave\
8 her more strength to reach for what she desired most - to attend that magnificent e\
9 vent alongside everyone else but not just only that
10
11 === Frequency and presence penalty -2.0 ===
12 , there lived a little girl named Lucy. She lived a very happy life with her family.\
13 She lived a very simple life. She lived a very happy life. She lived a very happy l\
14 ife. She lived a very happy life. She lived a very happy life. She lived a very happ\
15 y life. She lived a very happy life. She lived a very happy life. She lived a very h\
16 appy life. She lived a very happy life. She lived a very happy life. She lived a very
```
如您所见，第二个输出在一定程度上不断产生相同的输出，例如“She lived a very happy life”。

### 控制输出数量
如果你想有多个结果，你可以使用 n 参数。

以下示例将产生 2 个结果：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=5,
    n=2
)

print(next)
```
这是上面 Python 代码生成的输出示例：
```json
1 {
2 "choices": [
3 {
4 "finish_reason": "length",
5 "index": 0,
6 "logprobs": null,
7 "text": " there was a kind old"
8 },
9 {
10 "finish_reason": "length",
11 "index": 1,
12 "logprobs": null,
13 "text": ", there was a king"
14 }
15 ],
16 "created": 1674597690,
17 "id": "cmpl-6cLe6CssOiH4AYcLPz8eFyy53apdR",
18 "model": "text-davinci-003",
19 "object": "text_completion",
20 "usage": {
21 "completion_tokens": 10,
22 "prompt_tokens": 4,
23 "total_tokens": 14
24 }
25 }
```

### Getting the “best of”
可以要求 AI 模型为服务器端的给定任务生成可能的completions，并选择正确概率最高的任务。这可以使用 best_of 参数来完成。

使用 best_of 时，需要指定两个数字：n 和 best_of。

如前所述，n 是您希望看到的候选完成数。

注意：确保 best_of 大于 n。

让我们看一个例子：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=5,
    n=2,
    best_of=2
)

print(next)
```

### 控制 Completion 何时停止
在大多数情况下，阻止 API 生成更多文本很有用。

比方说，我们想要生成一个段落，而不是更多。在这种情况下，我们可以要求 API 在出现新行 (\n) 时停止补全文本。这可以使用如下类似的代码来完成：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=5,
    stop=["\n"]
)

print(next)
```
停止参数最多可以包含四个停止词。请注意，completion 不会在结果中包含停止序列。

这是另一个例子：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Once upon a time",
    max_tokens=5,
    stop=["\n", "Story", "End", "Once upon a time"]
)

print(next)
```

### Using Suffix After Text Completion（文本完成后使用后缀）
想象一下，我们想要创建一个 Python 字典，其中包含 0 到 9 之间的主要数字列表：
```py
{
    "primes"：[2, 3, 5, 7]
}
```
在这种情况下，API 应该返回 2, 3, 5, 7 。在这种情况下，我们可以使用后缀参数。该参数配置插入文本完成后的后缀。

让我们尝试两个例子来更好地理解。

在第一个示例中，我们将告诉 API 字典应该如何开始使用 {\n\t\"primes\"：[：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a JSON containing priary numbers between 0 and 9 \n\n{\n\t\"prim\es\": [",
)
print(next)
```
API 应返回此文本：

    2, 3, 5, 7]\n

如您所见，它关闭了列表和字典。

在第二个示例中，我们不希望看到 API 通过在结果文本的末尾插入 ]\n} 来关闭数据结构。这是使用后缀参数的时候：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a JSON containing priary numbers between 0 and 9 \n\n{\n\t\"prim\es\": [",
    suffix= "]\n}"
)
print(next)
```
API 现在应该返回：

    2, 3, 5, 7
    而不是之前的 (2, 3, 5, 7]\n})
这就是我们如何通知 API 有关完成后缀的信息。

### 示例：提取关键字
在这个例子中，我们要从文本中提取关键词。

这是我们将要使用的文本：
```text
The first programming language to be invented was Plankalkül, which was designed by 
Konrad Zuse in the 1940s, but not publicly known until 1972 (and not implemented until 1998). The first widely known and successful high-level programming language was Fortran, developed from 1954 to 1957 by a team of IBM researchers led by John Backus. The success of FORTRAN led to the formation of a committee of scientists to develop a "universal" computer language; the result of their effort was ALGOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with origins in academiato be successful. With the success of these initial efforts, programming languages became an active topic of research in the 1960s and beyond.
```
它取自维基百科¹⁹。

这是我们将要使用的代码：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

prompt = "The first programming language to be invented was Plankalkül, which was de\
signed by Konrad Zuse in the 1940s, but not publicly known until 1972 (and not imple\
mented until 1998). The first widely known and successful high-level programming l\
anguage was Fortran, developed from 1954 to 1957 by a team of IBM researchers led \
by John Backus. The success of FORTRAN led to the formation of a committee of scie\
ntists to develop a universal computer language; the result of their effort was AL\
GOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with o\
rigins in academia to be successful. With the success of these initial efforts, prog\
ramming languages became an active topic of research in the 1960s and beyond\n\nKeyw\
ords:"

tweet = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.5,
    max_tokens=300,
)

print(tweet)
```
此处的 prompt 如下所示：
```text
1 The first programming language to be invented was Plankalkül
2 [...]
3 [...]
4 in the 1960s and beyond
5
6 Keywords:
By appending “keywords” to a new line in the prompt, the model will recognize that we need
keywords and the output should look something like this:
1 programming language, Plankalkül, Konrad Zuse, FORTRAN, John Backus, ALGOL 58, John \
2 McCarthy, MIT, Lisp
You can play with the prompt and try different things such as:
1 The first programming language to be invented was Plankalkül
2 [...]
3 [...]
4 in the 1960s and beyond
5
6 Keywords:
7 -
```
这转化为：
```text
prompt = "The first programming language to be invented was Plankalkül, which was de\
signed by Konrad Zuse in the 1940s, but not publicly known until 1972 (and not imple\
mented until 1998). The first widely known and successful high-level programming l\
anguage was Fortran, developed from 1954 to 1957 by a team of IBM researchers led \
by John Backus. The success of FORTRAN led to the formation of a committee of scie\
ntists to develop a universal computer language; the result of their effort was AL\
GOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with o\
rigins in academia to be successful. With the success of these initial efforts, prog\
ramming languages became an active topic of research in the 1960s and beyond\n\nKeyw\
ords: \n-"
```
在这种情况下，我们应该得到与以下类似的结果：
```text
1 Plankalkül
2 - Konrad Zuse
3 - FORTRAN
4 - John Backus
5 - IBM
6 - ALGOL 58
7 - John McCarthy
8 - MIT
9 - Lisp
```

### 示例：生成推文
我们将继续使用前面的示例，但我们附加推文而不是关键字。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

prompt = ""The first programming language to be invented was Plankalkül, which was designed by Konrad Zuse in the 1940s, but not publicly known until 1972 (and not imple\
mented until 1998). The first widely known and successful high-level programming l\
anguage was Fortran, developed from 1954 to 1957 by a team of IBM researchers led \
by John Backus. The success of FORTRAN led to the formation of a committee of scie\
ntists to develop a universal computer language; the result of their effort was AL\
GOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with o\
rigins in academia to be successful. With the success of these initial efforts, prog\
ramming languages became an active topic of research in the 1960s and beyond\n\nTwee\
t:"

tweet = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.5,
    max_tokens=300,
)

print(tweet)
```
可以看到，提示是：
```text
1 The first programming language to be invented was Plankalkül
2 [...]
3 [...]
4 in the 1960s and beyond
5
6 Tweet:
```
这是一个输出示例：
```text
The first programming language was Plankalk\u00fcl, invented by Konrad Zuse in the 1940s.
```
我们还可以生成推文并使用如下提示提取主题标签：
```text
1 The first programming language to be invented was Plankalkül
2 [...]
3 [...]
4 in the 1960s and beyond
5
6 Tweet with hashtag:
```
这是代码的样子：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

prompt = ""The first programming language to be invented was Plankalkül, which was designed by Konrad Zuse in the 1940s, but not publicly known until 1972 (and not imple\
mented until 1998). The first widely known and successful high-level programming l\
anguage was Fortran, developed from 1954 to 1957 by a team of IBM researchers led \
by John Backus. The success of FORTRAN led to the formation of a committee of scie\
ntists to develop a universal computer language; the result of their effort was AL\
GOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with o\
rigins in academia to be successful. With the success of these initial efforts, prog\
ramming languages became an active topic of research in the 1960s and beyond\n\nTwee\
t with hashtags:"

tweet = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.5,
    max_tokens=300,
)

print(tweet)
```
结果应该是这样的：
```text
#Plankalkül was the first #programming language, invented by Konrad Zuse in the 1940s.
#Fortran, developed by John Backus and IBM in the 1950s, was the first widely known and successful high-level programming language. 
#Lisp, developed by John McCarthy of MIT in the 1960s, was the first language with origins in academia to be successful.
```
要调整结果，您可以使用 max_tokens 来更改推文的长度，因为您知道：

    • 100 个代币～= 75 个字
    • 推文不应超过280 个字符 
    • 英语中的平均单词为4.7 个字符。

https://platform.openai.com/tokenizer  可帮助您了解 API 如何标记一段文本以及其中的标记总数。

### 示例：生成说唱歌曲
在这个例子中，我们将看到如何生成说唱歌曲。您可以重复使用相同的示例来生成任何其他类型的文本。让我们看看如何完成：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

my_song = openai.Completion.create(
    model="text-davinci-002",
    prompt="Write a rap song:\n\n",
    max_tokens=200,
    temperature=0.5
)

print(my_song.choices[0]["text"].strip())
```
请注意我们如何使用 my_song.choices[0]["text"].strip() 直接访问和剥离文本以仅打印歌曲：
```text
1 I was born in the ghetto
2
3 Raised in the hood
4
5 My momma didn't have much
6
7 But she did the best she could
8
9 I never had much growing up
10
11 But I always had heart
12
13 I knew I was gonna make it
14
15 I was gonna be a star
16
17 Now I'm on top of the world
18
19 And I'm never looking back
```
您可以发挥模型和其他参数的“创造力”，测试、调整和尝试不同的组合。


### 示例：生成待办事项列表
在这个例子中，我们要求模型生成一个在美国创建公司的待办事项列表。我们需要清单上的五项。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-002",
    prompt="Todo list to create a company in US\n\n1.",
    temperature=0.3,
    max_tokens=64,
    top_p=0.1,
    frequency_penalty=0,
    presence_penalty=0.5,
    stop-["6."],
)

print(next)
```
应类似于以下内容：
```text
1. Choose a business structure.

2. Register your business with the state.

3. Obtain a federal tax ID number.

4. Open a business bank account.

5. Establish business cred
```
在我们的代码中，我们使用了以下参数：
```text
1 model="text-davinci-002"
2 prompt="Todo list to create a company in US\n\n1."
3 temperature=0.3
4 max_tokens=64
5 top_p=0.1
6 frequency_penalty=0
7 presence_penalty=0.5
8 stop=["6."]
```
让我们一一重新审视它们：

model：指定 API 应该用于生成文本完成的模型。在这种情况下，它正在使用“text-davinci-002”。

prompt：是 API 用作生成完成的起点的文本。在我们的例子中，我们使用的提示是在美国创建公司的待办事项列表。第一项应该以“1.”开头，知道我们要求的输出应该是这种格式:
```
1. <1st item>
2. <2nd item>
3. <3nd item>
4. <4th item>
5. <5th item>
```
temperature 控制着模型生成的文本的“创造力”。温度越高将更有创意和多样化的完成。另一方面，较低的温度将导致更“保守”和可预测的完成。在这种情况下，温度设置为 0.3。

max_tokens 限制 API 将生成的最大令牌数。在我们的例子中，令牌的最大数量为 64。您可以增加此值，但请记住，您生成的令牌越多，您需要支付的积分就越多。在学习和测试时，保持较低的值将帮助您避免超支。

top_p 控制 API 在生成时考虑的分布质量的比例下一个令牌。较高的值将导致更保守的补全，而较低的值将导致更多样化的补全。在这种情况下，top_p 设置为 0.1。不建议同时使用这个和温度，但这也不是一个阻塞问题。

frequency_penalty 用于调整模型对生成频繁词或稀有词的偏好。正值会减少出现频繁词的机会，而负值会增加出现频率的机会。在这种情况下，frequency_penalty 设置为 0

presence_penalty 用于调整模型对生成存在或提示中不存在。正值将减少提示中出现单词的机会，负值将增加它们。在我们的示例中，presence_penalty 设置为 0.5。

stop 用于指定 API 应在其后停止生成完成的标记序列。在在我们的例子中，因为我们只想要 5 个项目，所以我们应该在令牌 6. 生成后停止生成。

### 结论
OpenAI Completions API 是用于在各种上下文中生成文本的强大工具。通过正确的参数和设置，它可以生成与任务相关的听起来自然的文本。

通过为一些参数（例如频率和存在惩罚）配置正确的值，可以定制结果以产生期望的结果。

通过控制完成何时停止的能力，用户还可以控制生成的文本的长度。这也可能有助于减少生成的代币数量并间接降低成本。



## 使用 GPT 编辑文本
在给出提示和一组指令后，您正在使用的 GPT 模型将接受提示，然后使用其算法生成原始提示的修改版本。

根据您的说明，此修改后的版本可能比初始提示更长和/或更详细。

GPT 模型能够理解提示的上下文和给定的说明，从而确定哪些附加细节最有利于包含在输出中。


### 示例：翻译文本
我们一直在使用名为“chatgptforpythondevelopers”的相同开发环境。

首先激活它：
```sh
workon chatgptforpythondevelo
``` 
请注意，“.env”文件应始终存在于 Python 文件所在的当前目录中。

现在创建包含以下代码的“app.py”文件：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    input="Hallo Welt",
    instruction="Translate to English",
)

print(response)
```
在上面的代码中，我们要求 API 将德语文本翻译成英语:
```py
response = openai.Edit.create(
    model="text-davinci-002",
    input="Hallo Welt",
    instruction="Translate to English",
)

print(response)
```
执行上面的代码后，您应该会看到以下输出：
```json
1 {
2 "choices": [
3 {
4 "index": 0,
5 "text": "Hello World!\n"
6 }
7 ],
8 "created": 1674653013,
9 "object": "edit",
10 "usage": {
11 "completion_tokens": 18,
12 "prompt_tokens": 20,
13 "total_tokens": 38
14 }
15 }
```
API 返回了一个索引为 0 的选项，Hello World!\\n，如上面的输出所示。

与 Completion 不同，我们为 API 提供提示，我们需要在此处提供说明和输入。

### 指令是必需的，输入是可选的
重要的是要注意指令是必需的，而输入是可选的。

可以只提供指令并在同一行中包含输入：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instruction="Translate the following sentence to English: 'Hallo Welt'",
)

print(response)
```

### 使用Completions Endpoint进行编辑和互相转换（Editing Using the Completions Endpoint and Vice Versa）
您可以使用 edits endpoint 执行的一些任务，也可以通过使用 completions endpoint来完成。您可以选择最适合您的需求。

这是使用编辑端点的翻译任务示例：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instruction="Translate from English to French, Arabic, and Spanish.",
    input="The cat sat on the mat."
)

print(response)
```
这与上面的任务相同，但使用completion endpoints：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

next = openai.Completion.create(
    model="text-davinci-002",
    prompt="""
    Translate the follwing sentence from English to French, Arabic, and Spanish.
    English: The cat sat on the mat.
    French:
    Arabic:
    Spanish:
    """,
    max_tokens=60,
    temperature=0
)

print(next)
```
另一方面，以下示例将使用 edits 端点来执行文本完成：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instruction="Complete the story",
    input="Once upon a time."
)

print(response['choices'][0]['text'])
```

### 格式化输出
让我们举个例子：我们要求edits 端点向 Golang 代码添加注释。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instruction="Explain the following Golang code:",
    input="""
package main

import (
    "io/ioutil"
    "log"
    "net/http"
)

func main(){
    resp, err := http.Get("https://website.com")
    if err != nil {
        log.Fatalln(err)
    }

    sb := string(body)
    log.Printf(sb)
}
    """
)

print(response['choices'][0]['text'])
```
输出将不可读：
```json
1 {
2 "choices": [
3 {
4 "index": 0,
5 "text": "\npackage main\n\nimport (\n\t// code to work with input and output\n\
6 \t\"io/ioutil\"\n\n\t// code to work with logging\n\t\"log\"\n\n\t// code to work wi\
7 th http\n\t\"net/http\"\n)\n\nfunc main() {\n resp, err := http.Get(\"https://webs\
8 ite.com\")\n if err != nil {\n log.Fatalln(err)\n }\n\n body, err := iout\
9 il.ReadAll(resp.Body)\n if err != nil {\n log.Fatalln(err)\n }\n\n sb := \
10 string(body)\n log.Printf(sb)\n} \n \n"
11 }
12 ],
13 "created": 1674765343,
14 "object": "edit",
15 "usage": {
16 "completion_tokens": 467,
17 "prompt_tokens": 162,
18 "total_tokens": 629
19 }
20 }
```
但是，如果您只打印文本，则应正确设置格式。
```py
将 
   print(response['choices'][0]['text']) 
改为
   print(response["choices"][0]["text"])
```

### 创造力与明确定义的答案
与completion endpoints 相同，我们可以使用温度参数。您可以使用两个不同的温度来尝试这个示例，以查看输出的差异：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response_1 = openai.Edit.create(
    model="text-davinci-002",
    instrucation="correct the spelling mistakes:",
    input="The kuick brown fox jumps over the lazy dog and",
    temperature=0,
)

response_2 = openai.Edit.create(
    model="text-davinci-002",
    instrucation="correct the spelling mistakes:",
    input="The kuick brown fox jumps over the lazy dog and",
    temperature=0.9,
)

print("Temperature 0:")
print(response_1['choices'][0]['text'])
print("Temperature 0.9:")
print(response_2['choices'][0]['text'])
```
通常，多次运行代码后，您可能会观察到第一次输出是一致的，而第二次输出从一次执行到下一次发生变化。对于拼写错误之类的用例，我们通常不需要创造力，因此将温度参数设置为 0 就足够了。

如果有其他用例，创意参数应该设置为大于 0，但这个不能。

这是一个更有创意的机会：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instrucation="Exerices is good for your health.",
    input="Edit the text to make it longer.",
    temperature=0.9,
)

print(response['choices'][0]['text'])
```
这是一个输出示例:
```
Exercise is good for your health. Especially if you haven't done any for a month
```
这是输出的另一种变体：
```
1 Exercise is good for your health.
2 It will help you improve your health and mood.
3 It is important for integrating your mind and bo
```
令人惊讶的是（或者可能不是），这是我在 temperature=0 时得到的输出：
```
1 Exercise is good for your health.
2 Exercise is good for your health.
3 Exercise is good for your health.
4 Exercise is good for your health.
5 Exercise is good for your health.
6 Exercise is good for your health.
7 ...
```
获得更多创意的另一种方法是使用 top_np 参数。

这是温度采样的替代方法，因此，建议不要同时使用两者temperature 和 top_p 同时存在:
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instrucation="Exerices is good for your health.",
    input="Edit the text to make it longer.",
    top_p=0.1,
)

print(response['choices'][0]['text'])
```
在上面的示例中，我使用了 top_p=0​​.1，这意味着该模式将考虑 top_p 概率质量 = 0.1 的标记的结果。换句话说，结果中只考虑包含前 10% 概率质量的标记。

### 生成多个编辑
在之前的所有示例中，我们始终只有一次编辑。但是，使用参数 n，可以获得更多。只需使用您想要的编辑次数：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    
    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Edit.create(
    model="text-davinci-002",
    instrucation="Exerices is good for your health.",
    input="Edit the text to make it longer.",
    top_p=0.2,
    n=2,
)

print(response['choices'][0]['text'])
print(response['choices'][1]['text'])
```
在上面的例子中，我使用 n=2 得到了两个结果。

我还使用了 top_p=0​​.2 。但是，这与结果的数量无关；我只是想获得更广泛的结果。



## 高级文本操作示例
到目前为止，我们已经了解了如何使用不同的端点：编辑和完成。让我们做更多的例子来理解模型提供的不同可能性。

### 链接完成和编辑
在此示例中，我们将要求模型从文本中生成一条推文，然后对其进行翻译。

在第一个任务中，我们将使用完成端点获取推文的文本，然后是第二个任务的代码，即翻译推文：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

prompt = "The first programming language to be invented was Plankalkül, which was de\
signed by Konrad Zuse in the 1940s, but not publicly known until 1972 (and not imple\
mented until 1998). The first widely known and successful high-level programming l\
anguage was Fortran, developed from 1954 to 1957 by a team of IBM researchers led \
by John Backus. The success of FORTRAN led to the formation of a committee of scie\
ntists to develop a universal computer language; the result of their effort was AL\
GOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with o\
rigins in academia to be successful. With the success of these initial efforts, prog\
ramming languages became an active topic of research in the 1960s and beyond\n\nTweet with hashtags:"

english_tweet = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.5,
    max_tokens=20,
)

english_tweet_text = english_tweet["choices"][0]["text"].strip()
print("English Tweet:")
print(english_tweet_text)

spanish_tweet = openai.Edit.create(
    model="text-davinci-edit-001",
    input=english_tweet_text,
    instruction="Translate to Spanish",
    temperature=0.5,
)

spanish_tweet_text = spanish_tweet["choices"][0]["text"].strip()
print("Spanish Tweet:")
print(spanish_tweet_text)
```
通过执行上面的代码，我们可以看到两种不同语言的两条推文:
```
English Tweet:
  The #first #programming #language to be invented was #Plankalkül
Spanish Tweet:
  El primer lenguaje de programación inventado fue #Plank
```
请注意，我们添加了 strip() 以删除前导和尾随空格。

### Apple the Company vs. Apple the Fruit（上下文填充）
让我们创建一个代码来告诉我们一个词是名词还是形容词。当我们为模型提供诸如“光”之类的词时，可以看出其中的挑战之一。 “光”可以是名词、形容词或动词。
```
• 指示灯为红色。
• 这张桌子很轻。
• 你照亮了我的人生。
```
还有其他词可以同时用作名词、形容词或动词：“firm”、“fast”、“well”等，这个例子也适用于它们。

现在，如果我们想询问模型，我们可以编写以下代码：
```py
prompt = "Determine the part of speech of the word 'light'.\n\n"

result = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=20,
    temperature=1,
)
print(result.choices[0]["text"].strip())
```
你可以试试代码，它有时会输出动词，有时是形容词，有时是名词，但我也有这个“光”这个词可以用作名词、形容词或动词。

通过使用上下文，我们可以影响模型响应。上下文是给予模型的提示，以指导它完成用户定义的模式。

让我们在给它一些提示的同时询问模型。提示可以是任何有助于模型理解上下文的东西。

例如：
```py
prompt_a = "The light is red. Determine the part of speech of the word 'light'.\n\n"
prompt_b = "This desk is very light. Determine the part of speech of the word'light'.\n\n"
prompt_c = "You light up my life. Determine the part of speech of the word 'light'\n\n"

for prompt in [prompt_a, prompt_b, prompt_c]:
    result = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=20,
    temperature=0,
)

    print(result.choices[0]["text"].strip())
```
更好地理解上下文会导致以下结果：
```
1 noun(名词)
2 adjective（形容词）
3 verb（动词）
```
另一个例子是下面的例子，我们给模型两个不同的提示。在第一种情况下，我们想了解 Apple 是一家公司，而在第二种情况下，Apple 应该指的是水果。
```py
prompt = "Huawei:\ncompany\n\nGoogle:\ncompany\n\nMicrosoft:\ncompany\n\nApple:\n"
prompt="Huawei:\ncompany\n\nGoogle:\ncompany\n\nMicrosoft:\ncompany\n\nApricot:\nF\ruit\n\nApple:\n"

result = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=20,
    temperature=0,
    stop=["\n", " "],
)

print(result.choices[0]["text"].strip())
```

### 基于用户定义的模式获取加密货币信息（上下文填充）
让我们看第二个例子，我们提供模型在输出中应遵循的模式或模板。

我们的目标是获取有关给定加密货币的一些信息，包括其简称、创建日期、其 Coingecko 页面以及历史最高价和最低价。
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

prompt = """Input: Bitcoin
Output:
BTC was created in 2008, you can learn more about it here: https://bitcoin.org/en/ and get the latest price here: https://www.coingecko.com/en/coins/bitcoin.\
It's all-time high is $64,895.00 and it's all-time low is $67.81.

Input: Ethereum
Output:
ETH was created in 2015, you can learn more about it here: https://ethereum.org/en/\
and get the latest price here: https://www.coingecko.com/en/coins/ethereum
It's all-time high is $4,379.00 and it's all-time low is $0.43.

Input: Dogecoin
Output:
DOGE was created in 2013, you can learn more about it here: https://dogecoin.com/ and get the latest price here: https://www.coingecko.com/en/coins/dogecoin\
It's all-time high is $0.73 and it's all-time low is $0.000002.

Input: Cardano
Output:\n"""

result = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=200,
    temperature=0,
)

print(result.choices[0]["text"].strip())
```
我们首先给出模型应该返回什么的例子：
```
Input: BTC was created in 2008, you can learn more about it here: https://bitcoin.org/en/ and get the latest price here: https://www.coingecko.com/en/coins/bitcoin.
It's all-time high is $64,895.00 and it's all-time low is $67.81.

Input: Ethereum
Output:
ETH was created in 2015, you can learn more about it here: https://ethereum.org/en/ 
and get the latest price here: https://www.coingecko.com/en/coins/ethereum
It's all-time high is $4,379.00 and it's all-time low is $0.43.

Input: Dogecoin
Output:
DOGE was created in 2013, you can learn more about it here: https://dogecoin.com/ and get the latest price here: https://www.coingecko.com/en/coins/dogecoin
It's all-time high is $0.73 and it's all-time low is $0.000002.
```
然后我们调用端点。您可以根据需要更改输出格式。例如，如果您需要 HTML 输出，您可以将 HTML 标记添加到答案中。
例如：
```
Input: Bitcoin
Output:
BTC was created in 2008, you can learn more about it <a href="https://bitcoin.org/en/">here</a> and get the latest price <a href="https://www.coingecko.com/en/coins/bitcoin">here</a>.
It's all-time high is $64,895.00 and it's all-time low is $$67.81.
```
该模型将返回类似的输出：
```
Cardano was created in 2015, you can learn more about it <a href="https://www.cardano.org/en/home/">here</a> and get the latest price <a href="https://www.coingecko.com/en/coins/cardano">here</a>.
It's all-time high is $1.33 and it's all-time low is $0.000019.Let’s make it reusable with other cryptocurrencies:
```

### 创建聊天机器人助手以帮助处理 Linux 命令
免责声明：这部分的灵感来自 2020 年 OpenAI 的一个旧演示。

我们的目标是开发一个命令行工具，可以通过对话来帮助我们使用 Linux 命令。

让我们从这个例子开始：
```py
import os
import openai

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

prompt = """
Input: List all the files in the current directory
Output: ls -l

Input: List all the files in the current directory, including hidden files
Output: ls -la

Input: Delete all the files in the current directory
Output: rm *

Input: Count the number of occurrences of the word "sun" in the file "test.txt"
Output: grep -o "sun" test.txt | wc -l
Input:{}
Output:
"""

result = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt.format("Count the number of files in the current directory"),
    max_tokens=200,
    temperature=0,
)

print(result.choices[0]["text"].strip())
```
我们需要来自模型的单一响应，这就是我们使用零温度的原因。我们正在为模型提供足够的令牌来处理输出。

模型提供的答案应该是：
```sh
ls -l |wc -l
```
我们将使用 click²¹ (CLI Creation Kit)，这是一个用于使用最少代码创建命令行界面的 Python 包。这将使我们的程序更具交互性。

激活虚拟开发环境后开始安装Python包：
```sh
workon chatgptforpythondevelopers
pip install click==8.1.3
```
然后让我们创建这个 [app.py](http://app.py) 文件：
```py
import os
import openai
import click

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

_prompt = """
Input: List all the files in the current directory
Output: ls -l

Input: List all the files in the current directory, including hidden files
Output: ls -la

Input: Delete all the files in the current directory
Output: rm *

Input: Count the number of occurrences of the word "sun" in the file "test.txt"
Output: grep -o "sun" test.txt | wc -l

Input: {}
Output:"""

while True:
    request = input(click.style("Input", fg="green"))
    prompt = _prompt.format(request)
    result = openai.Completion.create(
        model = "text-davinci-002",
        prompt=prompt,
        temprature=0.0,
        max_token=100,
        stop=["\n"],
    )

    command = result.choices[0].text.strip()
    click.echo(click.style("Output: ", fg="yellow") + command)
    click.echo()
```
我们使用相同的提示。我们所做的唯一更改是在无限循环内的代码中添加点击。执行 [app.py](http://app.py) 时，我们的程序会请求输出（请求），然后将其插入提示并将其传递给 API。

在程序的最后，单击打印结果。最后的 click.echo() 将打印一个空行。
```sh
$ python app.py

Input: list all
Output: ls

Input: delete all
Output: rm -r *

Input: count all files
Output: ls -l | wc -l

Input: count all directories
Output: find . -type d | wc -l

Input: count all files that are not directories
Output: find . ! -type d | wc -
```
让我们实现一个退出命令：
```py
import os
import openai
import click

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

_prompt = """
Input: List all the files in the current directory
Output: ls -l

Input: List all the files in the current directory, including hidden files
Output: ls -la

Input: Delete all the files in the current directory
Output: rm *

Input: Count the number of occurrences of the word "sun" in the file "test.txt"
Output: grep -o "sun" test.txt | wc -l

Input: {}
Output:"""

while True:
    request = input(click.style("Input", fg="green"))
    prompt = _prompt.format(request)

    if request == "exit":
        break

    result = openai.Completion.create(
        model = "text-davinci-002",
        prompt=prompt,
        temprature=0.0,
        max_token=100,
        stop=["\n"],
    )

    command = result.choices[0].text.strip()
    click.echo(click.style("Output: ", fg="yellow") + command)
    click.echo()
```
最后，让我们实现一条指令来执行生成的命令：
```py
import os
import openai
import click

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

_prompt = """
Input: List all the files in the current directory
Output: ls -l

Input: List all the files in the current directory, including hidden files
Output: ls -la

Input: Delete all the files in the current directory
Output: rm *

Input: Count the number of occurrences of the word "sun" in the file "test.txt"
Output: grep -o "sun" test.txt | wc -l

Input: {}
Output:"""

while True:
    request = input(click.style("Input", fg="green"))
    prompt = _prompt.format(request)

    if request == "exit":
        break

    result = openai.Completion.create(
        model = "text-davinci-002",
        prompt=prompt,
        temprature=0.0,
        max_token=100,
        stop=["\n"],
    )

    command = result.choices[0].text.strip()
    click.echo(click.style("Output: ", fg="yellow") + command)

    click.echo(click.style("Execute? (y/n): ", fg="yellow"), nl=False)
    choice = input()
    if choice == "y":
        os.system(command)
    elif choice == "n":
        continue
    else:
        click.echo(click.style("Invalid choice. Plase enter 'y' or 'n'.", fg="red"))

    click.echo()
```
现在，如果你执行 python app.py，你会看到它询问你是否要执行命令：
```sh
Input (type 'exit' to quit): list all files in /tmp
Output: ls /tmp
Execute? (y/n): y
<files in /tmp/ appears here>

Input (type 'exit' to quit): list all files in /tmp
Output: ls /tmp
Execute? (y/n): cool
Invalid choice. Please enter 'y' or 'n'.
```
我们还可以添加一个 try..except 块来捕获任何可能的异常：
```py
import os
import openai
import click

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

_prompt = """
Input: List all the files in the current directory
Output: ls -l

Input: List all the files in the current directory, including hidden files
Output: ls -la

Input: Delete all the files in the current directory
Output: rm *

Input: Count the number of occurrences of the word "sun" in the file "test.txt"
Output: grep -o "sun" test.txt | wc -l

Input: {}
Output:"""

while True:
    request = input(click.style("Input", fg="green"))
    prompt = _prompt.format(request)

    if request == "exit":
        break
    try:
        result = openai.Completion.create(
            model = "text-davinci-002",
            prompt=prompt,
            temprature=0.0,
            max_token=100,
            stop=["\n"],
        )

        command = result.choices[0].text.strip()
        click.echo(click.style("Output: ", fg="yellow") + command)

        click.echo(click.style("Execute? (y/n): ", fg="yellow"), nl=False)
        choice = input()
        if choice == "y":
            os.system(command)
        elif choice == "n":
            continue
        else:
            click.echo(click.style("Invalid choice. Plase enter 'y' or 'n'.", fg="red"))
    except Exception as e:
        click.echo(click.style("The command could not be executed. {}".format(e), fg="red"))
        pass

    click.echo()
```


## Embedding（嵌入）

### 嵌入概述
如果我们想用一句话来描述这个特性，我们会说 OpenAI 的文本嵌入衡量了两个文本字符串彼此之间的相似程度。

一般来说，嵌入通常用于诸如查找与搜索查询最相关的结果、根据文本字符串的相似程度将文本字符串分组在一起、推荐具有相似文本字符串的项目、查找与其他文本字符串非常不同的项目等任务，分析不同的文本字符串彼此之间有何不同，并根据它们最相似的内容来标记文本字符串。

从实用的角度来看，嵌入是一种将现实世界的对象和关系表示为向量（数字列表）的方式。相同的向量空间用于衡量两个事物的相似程度。

### 用例
OpenAI 的文本嵌入衡量文本字符串的相关性，可用于多种目的。

这些是一些用例：
```
• 自然语言处理(NLP) 任务，例如情感分析、语义相似性和情感分类。

• 为机器学习模型生成文本嵌入特征，例如关键字匹配、文档分类和主题建模。

• 生成与语言无关的文本表示，允许对文本字符串进行跨语言比较。

• 提高基于文本的搜索引擎和自然语言理解系统的准确性。

• 通过将用户的文本输入与范围广泛的文本字符串进行比较，创建个性化推荐。
```
我们可以将用例总结如下：
```
• 搜索：结果按与查询字符串的相关性排序
• 聚类：文本字符串按相似性分组
• 建议：推荐具有相关文本字符串的项目 
• 异常检测：识别相关性很小的异常值 
• 多样性测量：分析相似性分布 
• 分类：文本字符串按其最相似的标签分类
```
以下是使用嵌入的一些实用方法（不一定是 OpenAI 的）：

### 特斯拉
使用非结构化数据可能会很棘手——原始文本、图片和视频并不总能让从头开始创建模型变得容易。这通常是因为由于隐私限制很难获得原始数据，而且创建好的模型可能需要大量的计算能力、庞大的数据集和时间。

嵌入是一种从一个上下文（如汽车图像）中获取信息并在另一个上下文（如游戏）中使用它的方法。这称为迁移学习，它可以帮助我们在不需要大量真实数据的情况下训练模型。

特斯拉正在他们的自动驾驶汽车中使用这种技术。

### 日历人工智能
Kalendar AI 是一种销售推广产品，它使用嵌入以自动方式从包含 3.4 亿个人资料的数据集中将正确的销售宣传与正确的客户相匹配。

自动化依赖于客户资料嵌入和销售宣传之间的相似性来对最合适的匹配项进行排名。根据 OpenAI 的说法，与他们以前的方法相比，这将不需要的目标减少了 40-56%。

### Notion
Notion 是一种在线工作区工具，它通过利用 OpenAI Embedding的强大功能改进了其搜索功能。这种情况下的搜索超出了该工具当前使用的简单关键字匹配系统。

这一新功能使 Notion 能够更好地理解存储在其平台中的内容的结构、上下文和含义，使用户能够执行更精确的搜索并更快地找到文档。

### DALL·E 2
DALL·E 2 是一个将文本标签转换为图像的系统。

它通过使用称为 Prior 和 Decoder 的两个模型来工作。 Prior 采用文本标签并创建 CLIP²² 图像embeddings，而解码器采用 CLIP 图像embeddings并生成学习图像。然后图像从 64x64 放大到 1024x1024。

### 要求
要使用embedding，您应该使用以下命令安装数据库：
```sh
 pip install datalib
 # 在本指南的另一个层面，我们将需要 Matplotlib 和其他库：
 pip install  matplotlib plotly scipy scikit-learn
```
确保将它安装在正确的虚拟开发环境中。

该软件包还将安装 pandas 和 NumPy 等工具。

这些库是 AI 和数据科学中最常用的库。
```
• pandas 是一种快速、强大、灵活且易于使用的开源数据分析和操作工具，构建于 Python 之上。
• NumPy 是另一个 Python 库，它添加了对大型多维数组和矩阵的支持，以及用于对这些数组进行运算的大量高级数学函数集合。
• Matplotlib 是Python 编程语言及其数值数学扩展NumPy 的绘图库。
• plotly.py 是一个交互式的、开源的、基于浏览器的Python 图形库。
• SciPy 是一个免费的开源Python 库，用于科学计算和技术计算。
```

### 理解 Text Embedding
让我们从这个例子开始：
```py
import os
import openai
import click

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Embedding.create(
    model = "text-embedding-ada-002",
    input = "I am a programmer."
)

print(response)
```
像往常一样，我们正在导入 openai 、进行身份验证并调用端点。然而，这次我们使用“Ada”，这是 OpenAI 上唯一可用于嵌入的最佳模型。 OpenAI 团队建议在几乎所有用例中使用 text-embedding-ada-002，因为他们将其描述为“更好、更便宜且更易于使用”。

输出应该很长并且应该如下所示：
``` json
1 {
2 "data": [
3 {
4 "embedding": [
5 -0.0169205479323864,
6 -0.019740639254450798,
7 -0.011300412937998772,
8 -0.016452759504318237,
9 [..]
10 0.003966170828789473
11 -0.011714739724993706
12 ],
13 "index": 0,
14 "object": "embedding"
15 }
16 ],
17 "model": "text-embedding-ada-002-v2",
18 "object": "list",
19 "usage": {
20 "prompt_tokens": 4,
21 "total_tokens": 4
22 }
23 }
```
我们可以直接使用以下方式访问嵌入：
```py
print(response["embedding
```
我们编写的程序打印了一个浮点数列表，例如 0.010284645482897758 和0.013211660087108612.
这些浮点数表示由 OpenAI“text-embedding-ada-002”模型生成的输入文本“我是程序员”的嵌入。

嵌入是捕获其含义的输入文本的高维表示。这有时被称为向量表示或简称为嵌入向量。

嵌入是一种使用大数值表示对象（例如文本）的方式。每个值代表对象意义的一个特定方面以及该特定对象在该方面的强度。在文本的情况下，可以表示文本的主题、情感或其他语义特征。

换句话说，您在这里需要了解的是，嵌入端点生成的向量表示是一种以机器学习模型和算法可以理解的格式表示数据的方式。它是一种获取给定输入并将其转换为这些模型和算法可以使用的形式的方法。

我们将看到如何在不同的用例中使用它。

### 多个输入的Embedding
在上一个例子中，我们使用了：
```py
input = "I am a programmer."
```
可以使用多个输入，我们看看是怎么做到的：
```py
import os
import openai
import click

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

response = openai.Embedding.create(
    model = "text-embedding-ada-002",
    input = ["I am a programmer.", "I am a writer"],
)

for data in response["data"]:
    print(data["embedding"])
```
请务必注意，每个输入的长度不得超过 8192 个标记。

### 语义搜索
在本指南的这一部分，我们将使用 OpenAI 嵌入实现语义搜索。

这是一个基本示例，但我们将通过更高级的示例。

让我们从身份验证开始：
```py
import openai
import os
import pandas as pd
import numpy as np

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()
```
接下来，我们将创建一个名为“words.csv”的文件。 CSV 文件包含一个名为“文本”的列和一个随机单词列表：
```csv
1 text
2 apple
3 banana
4 cherry
5 dog
6 cat
7 house
8 car
9 tree
10 phone
11 computer
12 television
13 book
14 music
15 food
16 water
17 sky
18 air
19 sun
20 moon
21 star
22 ocean
23 desk
24 bed
25 sofa
26 lamp
27 carpet
28 window
29 door
30 floor
31 ceiling
32 wall
33 clock
34 watch
35 jewelry
36 ring
37 necklace
38 bracelet
39 earring
40 wallet
41 key
42 photo
```
在数据操作（包括 CSV 文件中的数据）方面，Pandas 是一个非常强大的工具。这完全适用于我们的用例。让我们使用 pandas 读取文件并创建一个 pandas 数据框。
```py
df = pd.read_csv('words.csv')
```
DataFrame 是最常用的 pandas 对象。

Pandas 官方文档将数据框描述为具有可能不同类型的列的二维标记数据结构。您可以将其视为电子表格或 SQL 表，或 Series 对象的字典。

如果你打印 df，你会看到这个输出：
```text
text
 0 apple
 1 banana
 2 cherry
 3 dog
 4 cat
 5 house
 6 car
 7 tree
 8 phone
 9 computer
10 television
11 book
12 music
13 food
14 water
15 sky
16 air
17 sun
18 moon
19 star
20 ocean
21 desk
22 bed
23 sofa
24 lamp
25 carpet
26 window
27 door
28 floor
29 ceiling
30 wall
31 clock
32 watch
33 jewelry
34 ring
35 necklace
36 bracelet
37 earring
38 wallet
39 key
40 photo
```
接下来，我们将在数据框中获取每个作品的embedding。为此，我们不打算使用 openai.Embedding.create() 函数，而是使用 get_embedding。两者都会做同样的事情，但第一个将返回一个包含embeddings和其他数据的 JSON，而第二个将返回一个embeddings列表。第二个在数据框中使用更实用。

该函数的工作原理如下：
```py
get_embedding("hello", engine='text-embedding-ada-002')
# 将返回 [-0.02499537356197834, -0.019351257011294365, ..etc]
```
我们还将使用每个数据框对象都具有的函数 apply 。此函数（应用）将函数应用于数据框的轴。
```py
# 导入获取 embedding 的函数
from openai.embeddings_utils import get_embedding

# 获取数据框中每个单词的embedding
df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
```
现在我们有一个带有两个轴的数据框，一个是文本，另一个是embeddings。最后一个包含第一个轴上每个单词的embedding。

让我们将数据帧保存到另一个 CSV 文件：
```py
df.to_csv('embeddings.csv')
```
它包含 3 列：id、text 和 embeddings。

现在让我们读取新文件并将最后一列转换为 numpy 数组。为什么？

因为在下一步中，我们将使用 cosine_similarity 函数。此函数需要一个 numpy 数组，而默认情况下它是一个字符串。

但是为什么不使用常规的 Python 数组/列表，而使用numpy数组呢？

实际上，numpy 数组在数值计算中被广泛使用。常规列表模块不为此类计算提供任何帮助。除此之外，数组消耗的内存更少并且速度更快。这是因为数组是存储在连续内存位置的同类数据类型的集合，而 Python 中的列表是存储在非连续内存位置的异构数据类型的集合。

回到我们的代码，让我们将最后一列转换为一个 numpy 数组：
```py
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
```
现在我们将要求用户输入，读取它，并使用 cosine_similarity 执行语义搜索相似度：
```py
# 获取用户的搜索词
user_search = input('Enter a search term: ')

# 获取搜索词的 Embedding
user_search_embedding = get_embedding(user_search, engine='text-embedding-ada-002')

# 导入 cosine_similari 计算余弦相似度
from openai.embeddings_utils import cosine_similarity

# 计算搜索词与dataf中每个词的余弦相似度
df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, user_search_embedding))
```
让我们看看上面代码中的 3 个最新操作是做什么的：
```
1、
user_search_embedding = get_embedding(user_search, engine='text-embedding-ada-002')
这行代码使用 get_embedding 函数来获取用户指定的搜索词 user_search 的embeddings。
引擎参数设置为“text-embedding-ada-002”，它指定要使用的 OpenAI 文本嵌入模型。
2、from openai.embeddings_utils import cosine_similarity
这行代码从 openai.embeddings_utils 模块导入 cosine_similarity 函数。
cosine_similarity 函数计算两个嵌入之间的余弦相似度。
3、df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, user_search_embedding))
这行代码在名为“similarity”的数据框中创建一个新列，并使用带有 lambda 函数的 apply 方法来计算用户搜索词的嵌入与数据框中每个词的嵌入之间的余弦相似度。
每对嵌入之间的余弦相似度存储在新的相似度列中。
```
下面是整体代码：
```py
import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

df = pd.read_csv('words.csv')

# 获取数据框中每个单词的embedding
df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

df.to_csv('embeddings.csv')

df = pd.read_csv('embeddings.csv')

df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# 获取用户的搜索词
user_search = input('Enter a search term: ')

# 获取搜索词的 Embedding
user_search_embedding = get_embedding(user_search, engine='text-embedding-ada-002')

# 计算搜索词与dataf中每个词的余弦相似度
df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, user_search_embedding))

print(df)
```
运行代码后，我输入了术语“office”，通过使用“相似度”轴，我们可以看到哪个词在语义上与“office”相似。浮点值越高，“文本”列中的词越相似。

“项链”、“手镯”和“耳环”等词的得分为 0.77，而“桌子”等词的得分为 0.88。

为了使结果更具可读性，我们可以按相似度轴对数据框进行排序：
```py
# 按相似度轴对数据框进行排序
df = df.sort_values(by='similarity', ascending=False)
```
我们还可以使用以下方法获得前 10 个相似点：
```py
df.head(10)
```
我们来看看最终的代码：
```py
import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

init_api()

df = pd.read_csv('words.csv')

# 获取数据框中每个单词的embedding
df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

df.to_csv('embeddings.csv')

df = pd.read_csv('embeddings.csv')

df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# 获取用户的搜索词
user_search = input('Enter a search term: ')

# 获取搜索词的 Embedding
user_search_embedding = get_embedding(user_search, engine='text-embedding-ada-002')

# 计算搜索词与dataf中每个词的余弦相似度
df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, user_search_embedding))

df = df.sort_values(by='similarity', ascending=False)

print(df.head(10))
```

### 余弦相似度
两个词之间的相似性是通过我们所说的余弦相似性来计算的。要使用它，无需了解数学细节，但如果您想更深入地研究该主题，可以阅读本指南的这一部分。否则，跳过它不会改变您对如何使用 OpenAI API 构建智能应用程序的理解。

余弦相似度是衡量两个向量相似程度的一种方法。它查看两个向量（线）之间的角度并进行比较。余弦相似度是向量之间夹角的余弦值。结果是介于 -1 和 1 之间的数字。如果向量相同，则结果为 1。如果向量完全不同，则结果为 -1。如果向量成 90 度角，则结果为 0。在数学术语中，这是等式：
```
$$
Similarity = (A.B) / (||A||.||B||)
$$
• A 和B 是向量 
• A.B 是将两组数字相乘的方法。它是通过将一组中的每个数字与另一组中的相同数字相乘，然后将所有这些乘积加在一起来完成的。
• ||一个||是向量A的长度。它是通过对向量A的每个元素的平方和取平方根来计算的。
```
让我们看看向量 A = [2,3,5,2,6,7,9,2,3,4] 和向量 B = [3,6,3,1,0,9,2,3,4,5 ].

这就是我们如何使用 Python 获得它们之间的余弦相似度：
```py
import numpy as np
from numpy.linalg import norm

# 定义两个向量(vectors)
A = np.array([2,3,5,2,6,7,9,2,3,4])
B = np.array([3,6,3,1,0,9,2,3,4,5 ])

# 打印向量
print("Vector A: {}".format(A))
print("Vector B: {}".format(B))

# 计算余弦相似度
cosine = np.dot(A,B)/(norm(A)*norm(B))

# 打印余弦相似度
print("Cosine Similarity between A and B: {}".format(cosine))
```
我们也可以使用 Python Scipy 编写相同的程序：
```py
import numpy as np
from numpy.linalg import norm

# 定义两个向量(vectors)
A = np.array([2,3,5,2,6,7,9,2,3,4])
B = np.array([3,6,3,1,0,9,2,3,4,5 ])

# 打印向量
print("Vector A: {}".format(A))
print("Vector B: {}".format(B))

# 计算余弦相似度
cosine = 1 - spatial.distance.cosine(A,B)

# 打印余弦相似度
print("Cosine Similarity between A and B: {}".format(cosine))
```
或者使用 Scikit-Learn：
```py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 定义两个向量(vectors)
A = np.array([2,3,5,2,6,7,9,2,3,4])
B = np.array([3,6,3,1,0,9,2,3,4,5 ])

# 打印向量
print("Vector A: {}".format(A))
print("Vector B: {}".format(B))

# 计算余弦相似度
cosine = cosine_similarity([A],[B])

# 打印余弦相似度
print("Cosine Similarity: {}".format(cosine[0][0]))
```


## 高级Embedding 示例
### 预测您喜欢的咖啡
我们在这一部分的目标是根据用户的输入为他们推荐最好的咖啡拼配。例如，用户输入“Ethiopia Dumerso”，程序发现“Ethiopia Dumerso”、“Ethiopia Guji Natural Dasaya”和“Organic Dulce de Guatemala”是最接近他们选择的混合物，输出将包含这三种混合物。

我们需要一个可以在 Kaggle 上下载的数据集。转到 Kaggle 并下载名为 simplified_coffee.csv²³ 的数据集。 （您需要创建一个帐户。）

数据集有 1267 行（混合）和 9 个特征：
```
• name（咖啡名称） 
• roaster（烘焙商名称） 
• roast（烘焙类型）
• loc_country（烘焙商所在的国家/地区） 
• origin（咖啡豆的产地）
• 100g_USD（每 100g 美元的价格） 
• rating（满分 100 分） 
• review_date（审查日期） 
• review（审查文本）
```
我们对该数据集感兴趣的是用户的评论。这些评论是从 www.coffeereview.com 上截取的。

当用户输入咖啡名称时，我们将使用 OpenAI Embeddings API 获取该咖啡评论文本的嵌入。然后，我们将计算输入的咖啡评论与数据集中所有其他评论之间的余弦相似度。余弦相似度得分最高的评论将与输入咖啡的评论最相似。然后，我们会将最相似的咖啡的名称打印给用户。

让我们一步步开始。

激活您的虚拟开发环境并安装 nltk：
```sh
pip install nltk
```
自然语言工具包，或更常见的 NLTK，是一套库和程序，用于用 Python 编程语言编写的英语符号和统计自然语言处理。在下一步中，您将了解我们如何使用它。

现在，使用您的终端，键入 python 以进入 Python 解释器。然后，键入以下命令：
```py
import nltk

nltk.download('stopwords')
nltk.download('punkt')
```
NLTK 带有许多语料库、玩具语法、经过训练的模型等。以上（停用词和 punkt）是我们此演示所需的唯一内容。如果你想下载所有这些，你可以使用nltk.download('all') 代替。您可以在此处找到完整的语料库列表²⁵。我们将创建 3 个函数：
```py
import openai
import os
import pandas as pd
import numpy as np
import nltk
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def preprocess_review(review):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stopwords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(review.lower())
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)
```
init_api 函数将从 .env 文件中读取 API 密钥和组织 ID，并设置环境变量。

download_nltk_data 函数将下载 punkt 和停用词语料库（如果尚未下载）。

preprocess_review 函数将小写、标记化、删除停用词和阻止评论文本。

该函数使用 nltk.word_tokenize() 函数将评论标记为单个单词。

然后它使用从 NLTK 的停用词语料库中使用 nltk.corpus.stopwords.words() 函数获得的停用词列表，从评论中删除停用词（常见词，如“the”、“a”、“and”等）

最后，它使用来自 NLTK 的 Porter 词干分析器对单词进行词干分析，使用nltk.stem.PorterStemmer() 函数。词干提取是将单词缩减为词根的过程形式。例如，单词“running”、“ran”和“run”都有相同的词干“run”。这很重要，因为它将帮助我们减少评论文本中唯一单词的数量，通过减少参数数量优化我们模型的性能，并降低与 API 调用相关的任何成本。

该函数使用 Python 字符串的 .join() 方法将词干化的单词连接回单个字符串。这是我们将用于生成嵌入的最终预处理评论文本。

接下来，添加以下代码：
```py
init_api()
download_nltk_data()
```
将初始化 OpenAI API 并使用已定义的函数下载必要的 NLTK 数据。主要是，该功能会检查您是否没有手动下载所需的数据，如果没有则下载。

然后我们需要读取用户输入：
```py
# 读取用户输入
input_coffee_name = input("Enter a coffee name: ")
```
接下来，我们将 CSV 文件加载到 Pandas DataFrame 中。请注意，我们在这里只读取前 50 行。您可以更改此代码并删除 nrows 以加载所有 CSV 数据集。
```py
# 将 CSV 文件加载到 Pandas DataFrame 中（目前只有前 50 行以加快速度启动演示并避免为过多的 API 调用付费）
df = pd.read_csv('simplified_coffee.csv', nrows=50)
```
然后我们将预处理所有预览文本：
```py
# 预处理评论文本：小写、分词、去除停用词和词干
df['preprocessed_review'] = df['review'].apply(preprocess_review)
```
现在，我们需要获取每条评论的嵌入：
```py
# 获取每个评论的嵌入
review_embeddings = []
for review in df['preprocessed_review']:
    review_embeddings.append(get_embedding(review, engine='text-embedding-ada-002'))
```
然后我们将获得输入咖啡名称的索引。如果咖啡名称不在我们的数据库中，我们将退出程序：
```py
# 获取输入咖啡名称的索引
try:
    input_coffee_index = df[df['name'] == input_coffee_name].index[0]
except:
    print("Sorry, we don't have that coffee in our database. Please try again.")
    exit()
```
input_coffee_index = df[df['name'] == input_coffee_name].index[0] 使用 Pandas 的 df[df['name'] == input_coffee_name] 获取包含输入的 DataFrame 行
咖啡名称。

例如，df[df['my_column'] == my_value] 从 DataFrame df 中选择行，其中 my_column 列中的值等于 my_value。

此代码返回一个新的 DataFrame，其中仅包含满足此条件的行。生成的 DataFrame 具有与原始 DataFrame df 相同的列，但只有满足条件的行。

例如，如果 df 是咖啡评论的 DataFrame（我们的例子），而 my_column 是“name”列，那么 df[df['name'] == 'Ethiopia Yirgacheffe'] 将返回一个新的 DataFrame，其中包含只有“埃塞俄比亚耶加雪菲”咖啡的评论。

接下来，我们使用 index[0] 来获取该行的索引。

index[0] 用于检索 df[df['name'] == input_coffee_name] 返回的结果过滤 DataFrame 的第一个索引。这是 input_coffee_index = df[df['name'] == input_coffee_-name].index[0] 做到的：
```
1. df['name'] == input_coffee_name 创建一个布尔掩码，对于“name”列等于 input_coffee_name 的行为 True，对于所有其他行为 False。

2. df[df['name'] == input_coffee_name] 使用这个布尔掩码来过滤 DataFrame 和返回一个新的 DataFrame，它只包含“name”列等于的行输入_咖啡_名称。

3. df[df['name'] == input_coffee_name].index 返回过滤后的 DataFrame 的索引标签。

4. index[0] 从结果索引标签中检索第一个索引标签。由于过滤后的 DataFrame 仅包含一行，因此这是该行的索引标签。
```
接下来，我们将计算输入咖啡评论与所有其他评论之间的余弦相似度：
```py
# 计算输入咖啡的评论与所有其他评论之间的余弦相似度
similarities = []
input_review_embedding = review_embeddings[input_coffee_index]
for review_embedding in review_embeddings:
    similarity = cosine_similarity(input_review_embedding, review_embedding)
    similarities.append(similarity)
```
cosine_similarity(input_review_embedding, review_embedding)是使用OpenAI的openai.embeddings_utils.cosine_similarity()函数计算余弦相似度
在输入咖啡的评论和当前评论之间。 （我们之前在前面的示例中使用过此功能。）

之后，我们将获得最相似评论的索引（不包括输入咖啡的评论本身）：
```py
# 获取最相似评论的索引（不包括输入咖啡的评论自己）
most_similar_indices = np.argsort(similarties)[-6:-1]
```
如果你以前使用过numpy，你肯定对argsort很熟悉，你可以跳过下面关于它是如何工作的详细解释。

np.argsort(similarities)[-6:-1] 使用 NumPy 的 argsort() 函数来获取与输入咖啡的评论最相似的前 5 条评论。

以下是对正在发生的事情的逐步细分：

argsort() 返回将按升序对相似性数组进行排序的索引。例如，如果相似度为 [0.8, 0.5, 0.9, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0]，则如下所示np.argsort(similarities)[-6:-1] 会起作用：
```
1. np.argsort(similarities) 将返回排序后的索引：[9, 8, 7, 6, 5, 4, 1, 3,0, 2]。数组根据相似度值排序：相似度[0] = 0.8、相似度[1] = 0.5、相似度[2] = 0.9 等。

2. np.argsort(similarities)[-6:-1] 将返回排序数组末尾的第 6 到第 2 个索引：[5, 4, 1, 3, 0]。
```
当调用 np.argsort(similarities) 时，它返回一个索引数组，用于对相似度数组按升序排列。换句话说，排序数组中的第一个索引将
对应于具有最小值的相似度元素，并且排序数组中的最后一个索引将对应于具有最大值的相似度元素。

在示例 [0.8, 0.5, 0.9, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0] 中，最小值（0.0）的索引为 9，第二小值（0.1）的索引为 8 ， 等等。排序后的索引数组为 [9, 8, 7, 6, 5, 4, 1, 3, 0, 2]。

然后，通过对数组进行切片以获取数组末尾的第 6 到第 2 个元素：[5、4、1、3、0]，使用这个排序的索引数组来获取前 5 个最相似评论的索引。这些指数对应于相似度降序排列的最相似评论。

你可能会问，为什么我们不使用 [-5:] ？

如果我们使用 np.argsort(similarities)[-5:] 而不是 np.argsort(similarities)[-6:-1]，我们会得到 5 个最相似的评论，包括输入咖啡的评论本身。我们将输入咖啡的评论本身从最相似的评论中排除的原因——向用户推荐已经尝试过的相同咖啡并不是很有帮助。通过使用 [-6:-1]，我们从切片中排除了第一个元素，它对应于输入咖啡的评论。

您可能会问的另一个问题是，为什么评论本身在相似度数组中？

当我们使用 get_embedding() 函数为每个评论创建嵌入时，评论本身被添加到 review_embeddings 数组中。

接下来，我们将获得最相似咖啡的名称：
```py
# 获取最相似咖啡的名称
similar_coffee_names = df.iloc[most_similar_indices]['name'].tolist()
```
df.iloc[most_similar_indices]['name'].tolist() 使用 Pandas 的 iloc[] 函数来获取最相似咖啡的名称。这是一个解释：df.iloc[most_similar_indices] 正在使用 iloc[] 来获取对应的 DataFrame 行到最相似的评论。例如，如果最相似的索引是 [3, 4, 0, 2]，df.iloc[most_-similar_indices] 将返回对应于第 4、5、1 和
第三个最相似的评论。

然后我们使用 ['name'] 来获取这些行的名称列。最后，我们使用 tolist() 将列转换为列表。这为我们提供了最相似咖啡名称的列表。

最后，我们将打印结果：
```py
print("The most similar coffees to {} are:".format(input_coffee_name))
for coffee_name in similar_coffee_names:
    print(coffee_name)
```
最终代码：
```py
import openai
import os
import pandas as pd
import numpy as np
import nltk
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def preprocess_review(review):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stopwords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(review.lower())
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

init_api()
download_nltk_data()

# 读取用户输入
input_coffee_name = input("Enter a coffee name: ")

# 将 CSV 文件加载到 Pandas DataFrame 中（目前只有前 50 行以加快速度启动演示并避免为过多的 API 调用付费）
df = pd.read_csv('simplified_coffee.csv', nrows=50)

# 预处理评论文本：小写、分词、去除停用词和词干
df['preprocessed_review'] = df['review'].apply(preprocess_review)

# 获取每个评论的嵌入
review_embeddings = []
for review in df['preprocessed_review']:
    review_embeddings.append(get_embedding(review, engine='text-embedding-ada-002'))

# 获取输入咖啡名称的索引
try:
    input_coffee_index = df[df['name'] == input_coffee_name].index[0]
except:
    print("Sorry, we don't have that coffee in our database. Please try again.")
    exit()

# 计算输入咖啡的评论与所有其他评论之间的余弦相似度
similarities = []
input_review_embedding = review_embeddings[input_coffee_index]
for review_embedding in review_embeddings:
    similarity = cosine_similarity(input_review_embedding, review_embedding)
    similarities.append(similarity)

# 获取最相似评论的索引（不包括输入咖啡的评论自己）
most_similar_indices = np.argsort(similarties)[-6:-1]

# 获取最相似咖啡的名称
similar_coffee_names = df.iloc[most_similar_indices]['name'].tolist()

print("The most similar coffees to {} are:".format(input_coffee_name))
for coffee_name in similar_coffee_names:
    print(coffee_name)
```

### 进行“更模糊”的搜索
该代码的一个潜在问题是用户必须输入数据集中存在的咖啡的确切名称。一些例子是：“Estate Medium Roast”，“Gedeb Ethiopia”..etc 这在现实生活中不太可能发生。用户可能会漏掉一个字符或一个词、输入错误的名称或使用不同的大小写，这将退出搜索并显示一条消息对不起，我们的咖啡机中没有那种咖啡数据库。请再试一次。

一种解决方案是执行更灵活的查找。例如，我们可以忽略大小写搜索包含输入咖啡名称的名称：
```
