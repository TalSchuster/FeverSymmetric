import sys
if len(sys.argv) < 2:
    lr = 1
else:
    lr = float(sys.argv[1])
optimizer = optim.Adam(list([x]), lr=lr)
epochs = 2000
score_0 = 0.0
score_1 = 0.0
score_2 = 0.0
for i in tqdm(range(epochs)):
    optimizer.zero_grad()
    loss = 0.0
    losses = []
    for word,freq,score,word_sentence in zip(words[:TOP_N],train_freqs[:TOP_N],train_scores[:TOP_N],train_word_sentences[:TOP_N]):
        a,b,c,d = get_word_loss(x, word, freq, score, word_sentence, train_sentence_label, threshold=0.3)
        score_0 += b
        score_1 += c
        score_2 += d
        losses.append(a)
    loss = sum(losses) + 1e-10*torch.norm(x)
    loss.backward()
    optimizer.step()
    for p in x:
        p.data.clamp_(0)
    print(loss.data)
    print(score_0, score_1, score_2)
    print(sum(x.data.numpy().tolist()))

print(score_0,score_1,score_2)

import pickle
with open('parameters_' + str(epochs) + '.pkl', 'wb') as f:
    pickle.dump(x.data.numpy().tolist(),f)

scores = 0
for claim,weight,label in zip(train_claims, x.data.numpy().tolist(), train_sentence_label):
    f1.write(claim + "\n")
    scores += weight
    f2.write(str(weight) + "\n")
f1.close()
f2.close()
