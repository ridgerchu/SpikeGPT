########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import math, os, sys, types, time, gc
import torch
from src.utils import TOKENIZER
import matplotlib.ticker as ticker
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()


########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
########################################################################################################

args.RUN_DEVICE = "cuda" # 'cuda' // 'cpu' (already fast)
args.FLOAT_MODE = "fp32" # fp16 (good for GPU, does not work for CPU) // fp32 (good for CPU) // bf16 (less accurate, but works for CPU)

# if args.RUN_DEVICE == "cuda":
#     os.environ["RWKV_RUN_BACKEND"] = 'nvfuser' # !!!BUGGY!!! wrong output
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0'. very useful for GPU/CPU fp32, but might be harmful for GPU fp16. please benchmark !!!

#For BookCorpus Pre-trained model
# TOKEN_MODE = "char"
# WORD_NAME = "vocab_book"
# UNKNOWN_CHAR = ' '
# vocab_size = 77

#For 216M OpenWebText Pre-trained model
TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
vocab_size = 50277

MODEL_NAME = 'SpikeGPT-216M'
n_layer = 18
n_embd = 768
ctx_len = 1024

args.MODEL_NAME = MODEL_NAME
args.n_layer = n_layer
args.n_embd = n_embd
args.ctx_len = ctx_len
args.vocab_size = vocab_size
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

# context = 'A'
#context = "\nIn the"
#context = 'Pinky \n The pink ghost’s AI is designed to ”feel” opposite of the red ghost’s behavior. Pinky actually attempts to get out in front of Pac-Man. This is accomplished by setting the target 4 tiles ahead of Pac-Man’s current location in the direction that Pac-Man is travelling. One exception to this is when Pac-Man is traveling up. Due to an overflow bug in the code, the calculation includes a left offset equal to the expected up offset.'
#context = '''Corporal Michael P. Goeldin was an unskilled laborer from Ireland when he enlisted in Company A in November 1860. Goldein survived the war. Corporal Patrick O’Neal, also from Ireland, first enlisted in 1854 and served with Company L, 3d U.S. Artillery, in Oregon. He returned to the East Coast and enlisted in the company in 1860. O’Neal served until 1874, when he was named superintendent of the National Cemetery at Willets Point, New York. Corporal Benjamin Browne was a shoemaker from Orange County, New York. In August 1862, he enlisted in the newly formed 124th New York Volunteers, and was one of sixty-one men who transferred into Company A that October. Browne reenlisted in the company in February 1864 while it was camped at Brandy Station. He returned to civilian life after completing his enlistment in 1867.
#On 10 June, Artificer William Collins was promoted to corporal, probably to fill a combat leadership void for the crossing of the James River. Collins’s service record does not reflect the qualities he demonstrated to earn this promotion, but he had obviously overcome some serious problems. Born in Sacketts Harbor, New York, Collins enlisted in the company in December 1853 at the age of twenty-two, and reenlisted in December 1858. Just a month before the war began in April 1861, Collins went ”over the hill” and was not caught until three years later. Returned to the company on 22 March 1864, he was tried'''
#context = 'Aaron loves mint chocolate cake, but he requires that it be paired with mini chocolate chips, so I threw some of those in between the layers. I also had a few Peppermint Jo Jos on hand so I crushed them up and threw some of those in along with some crushed meringue cookies because, why not? It’s a total smorgasbord of minty chocolate chippy cookie crunchy goodness. I didn’t measure how much of each topping I used, but after I tasted the finished product, I wish I had added more. You can add anything you want- crushed candy canes, peppermint bark, etc. And don’t be afraid to use a heavy hand. Texture = good.'
context = 'Prehistoric man sketched an incredible array of prehistoric beasts on the rough limestone walls of a cave in modern day France 36,000 years ago. Now, with the help of cutting-edge technology, those works of art in the Chauvet-Pont-d’Arc Cave have been reproduced to create the biggest replica cave in the world. The manmade cavern named the Caverne du Pont-d’Arc has been built a few miles from the original site in Vallon-Pont-D’arc in Southern France and contains 1,000 painstakingly-reproduced drawings as well as around 450 bones and other features...\n Cavemen and women sketched an incredible array of prehistoric beasts on the rough limestone walls of a cave 36,000 years ago and now a replica has been created (pictured)'
# context = '\nSugar:'
# context = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

# context = "\n深圳是" # test Chinese
# context = "\n東京は" # test Japanese

# ###### A good prompt for Q&A ######
# context = '''
# Questions & Helpful Answers
# Ask Research Experts
# Question:
# Can penguins fly?

# Full Answer:
# '''

# ###### A good prompt for chatbot ######
# context = '''
# The following is a conversation between a highly knowledgeable and intelligent AI assistant called Bot, and a human user called User. In the following interactions, User and Bot converse in natural language, and Bot always answer User's questions. Bot is very smart, polite and humorous. Bot knows a lot, and always tells the truth. The conversation begins.

# User: who is president of usa?

# Bot: It’s Joe Biden; he was sworn in earlier this year.

# User: french revolution what year

# Bot: It started in 1789, but it lasted 10 years until 1799.

# User: guess i marry who ?

# Bot: Only if you tell me more about yourself - what are your interests?

# User: wat is lhc

# Bot: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

# User:''' # type your question here

NUM_TRIALS = 999
LENGTH_PER_TRIAL = 333

TEMPERATURE = 1.5
top_p = 0.7
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

print(f'\nUsing {args.RUN_DEVICE.upper()}. Loading {MODEL_NAME}...')
from src.model_run import RWKV_RNN

model = RWKV_RNN(args)

print(f'\nOptimizing speed...')
#out, _ = model.forward([187], None, None, None)
# print(out)
gc.collect()
torch.cuda.empty_cache()

# input(0)

print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == "pile":
    assert tokenizer.tokenizer.decode([187]) == '\n'

########################################################################################################

if tokenizer.charMode:
    context = tokenizer.refine_context(context)
    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
else:
    ctx = tokenizer.tokenizer.encode(context)
src_len = len(ctx)
src_ctx = ctx.copy()

print("\nYour prompt has " + str(src_len) + " tokens.")
print(
    "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
)

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

init_state = None
init_out = None
state = None
mem1 = None
mem2 = None
out = None

for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print(("-" * 50) + '\n' + context, end="")

    time_ref = time.time_ns()
    ctx = src_ctx.copy()

    if TRIAL == 0:
        for i in range(src_len):
            x = ctx[: i + 1]
            if i == src_len - 1:
                init_out, init_state, mem1, mem2 = model.forward(x, init_state, mem1, mem2)
            else:
                init_state, mem1, mem2 = model.forward(x, init_state, mem1, mem2, preprocess_only=True)
        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    out_last = src_len
    for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
        x = ctx[: i + 1]
        x = x[-ctx_len:]

        if i == src_len:
            out = init_out.clone()
            state = init_state.clone()
        else:
            out, state, mem1, mem2 = model.forward(x, state, mem1, mem2)
        if DEBUG_DEBUG:
            print("model", np.array(x), "==>", np.array(out), np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
        if TOKEN_MODE == "pile":
            out[0] = -999999999  # disable <|endoftext|>

        ttt = tokenizer.sample_logits(
            out,
            x,
            ctx_len,
            temperature=TEMPERATURE,
            top_p_usual=top_p,
            top_p_newline=top_p_newline,
        )
        ttt = int(ttt)
        ctx += [ttt]

        if tokenizer.charMode:
            char = tokenizer.itos[ttt]
            print(char, end="", flush=True)
        else:
            char = tokenizer.tokenizer.decode(ctx[out_last:])
            if '\ufffd' not in char: # is valid utf8 string?
                print(char, end="", flush=True)
                out_last = i+1

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
    )

print(("-" * 50) + '\n')
