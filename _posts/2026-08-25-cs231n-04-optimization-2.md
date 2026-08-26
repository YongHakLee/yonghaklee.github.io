---
title: "04. Backpropagation, Intuitions"
description: "연쇄 법칙에 기반한 역전파의 직관적 이해, 계산 그래프와 게이트, 벡터화된 기울기 계산."
date: 2026-08-25 09:15:00 +0900
categories: [Computer Vision, cs231n]
tags: [study, computer vision, cs231n, deep learning]
math: true
---

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **원문**: [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
> — CS231n: Convolutional Neural Networks for Visual Recognition (Stanford University) · © 2015 Andrej Karpathy, MIT License
>
> 원문을 문단 단위로 인용하고 그 아래에 한국어 번역을 붙였다. 인용 블록이 원문, 그 아래 문단이 번역이며, `역주` 박스와 `보충` 섹션은 원문에 없는 추가 내용이다.
{: .prompt-info }
<!-- markdownlint-restore -->

> Table of Contents:

목차는 다음과 같다.

> - [Introduction](#intro)
> - [Simple expressions, interpreting the gradient](#grad)
> - [Compound expressions, chain rule, backpropagation](#backprop)
> - [Intuitive understanding of backpropagation](#intuitive)
> - [Modularity: Sigmoid example](#sigmoid)
> - [Backprop in practice: Staged computation](#staged)
> - [Patterns in backward flow](#patterns)
> - [Gradients for vectorized operations](#mat)
> - [Summary](#summary)

- [들어가며](#intro)
- [간단한 식과 기울기의 해석](#grad)
- [복합 식, 연쇄 법칙, 역전파](#backprop)
- [역전파의 직관적 이해](#intuitive)
- [모듈화: sigmoid 예제](#sigmoid)
- [실전에서의 역전파: 단계별 계산](#staged)
- [역방향 흐름에 나타나는 패턴](#patterns)
- [벡터화된 연산의 기울기](#mat)
- [정리](#summary)

<a id="intro"></a>

### Introduction

> **Motivation**. In this section we will develop expertise with an intuitive understanding of **backpropagation**, which is a way of computing gradients of expressions through recursive application of **chain rule**. Understanding of this process and its subtleties is critical for you to understand, and effectively develop, design and debug neural networks.

**동기.** 이 절에서는 **역전파**를 직관적으로 이해하는 데 익숙해진다. 역전파는 **연쇄 법칙**을 재귀적으로 적용해 식의 기울기를 계산하는 방법이다. 이 과정과 그 미묘한 지점들을 이해해두는 것은 신경망을 이해하는 데도, 신경망을 효과적으로 개발하고 설계하고 디버깅하는 데도 결정적이다.

> **Problem statement**. The core problem studied in this section is as follows: We are given some function $$f(x)$$ where $$x$$ is a vector of inputs and we are interested in computing the gradient of $$f$$ at $$x$$ (i.e. $$\nabla f(x)$$ ).

**문제 설정.** 이 절에서 다루는 핵심 문제는 이렇다. 어떤 함수 $$f(x)$$가 주어져 있고 $$x$$는 입력 벡터인데, 우리는 $$x$$에서의 $$f$$의 기울기(즉 $$\nabla f(x)$$)를 계산하려 한다.

> **Motivation**. Recall that the primary reason we are interested in this problem is that in the specific case of neural networks, $$f$$ will correspond to the loss function ( $$L$$ ) and the inputs $$x$$ will consist of the training data and the neural network weights. For example, the loss could be the SVM loss function and the inputs are both the training data $$(x_i,y_i), i=1 \ldots N$$ and the weights and biases $$W,b$$. Note that (as is usually the case in Machine Learning) we think of the training data as given and fixed, and of the weights as variables we have control over. Hence, even though we can easily use backpropagation to compute the gradient on the input examples $$x_i$$, in practice we usually only compute the gradient for the parameters (e.g. $$W,b$$) so that we can use it to perform a parameter update. However, as we will see later in the class the gradient on $$x_i$$ can still be useful sometimes, for example for purposes of visualization and interpreting what the Neural Network might be doing.

**동기.** 우리가 이 문제에 관심을 두는 가장 큰 이유는, 신경망이라는 구체적인 경우에 $$f$$가 손실 함수($$L$$)에 해당하고 입력 $$x$$가 학습 데이터와 신경망 가중치로 이루어지기 때문이다. 예를 들어 손실은 SVM 손실 함수일 수 있고, 입력은 학습 데이터 $$(x_i,y_i), i=1 \ldots N$$과 가중치·편향 $$W,b$$ 둘 다이다. (기계 학습에서 대개 그렇듯) 학습 데이터는 주어진 채 고정된 것으로 보고 가중치는 우리가 조절할 수 있는 변수로 본다는 점에 유의하자. 그래서 역전파로 입력 예제 $$x_i$$에 대한 기울기도 얼마든지 계산할 수 있지만, 실전에서는 보통 매개변수(예컨대 $$W,b$$)에 대한 기울기만 계산해 매개변수 갱신에 쓴다. 다만 수업 뒤쪽에서 보겠지만 $$x_i$$에 대한 기울기도 때로는 쓸모가 있다. 예컨대 신경망이 무엇을 하고 있는지 시각화하고 해석하는 용도다.

> If you are coming to this class and you’re comfortable with deriving gradients with chain rule, we would still like to encourage you to at least skim this section, since it presents a rarely developed view of backpropagation as backward flow in real-valued circuits and any insights you’ll gain may help you throughout the class.

연쇄 법칙으로 기울기를 유도하는 데 이미 익숙한 채로 이 수업에 왔더라도 이 절만큼은 훑어보기를 권한다. 역전파를 실숫값 회로 위의 역방향 흐름으로 보는, 흔히 다루지 않는 관점을 제시하기 때문이다. 여기서 얻은 통찰은 수업 내내 도움이 될 것이다.

<a id="grad"></a>

### Simple expressions and interpretation of the gradient

> Lets start simple so that we can develop the notation and conventions for more complex expressions. Consider a simple multiplication function of two numbers $$f(x,y) = x y$$. It is a matter of simple calculus to derive the partial derivative for either input:
>
> $$
> f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x
> $$

더 복잡한 식으로 가기 전에 표기와 관례를 잡아둘 수 있도록 간단한 것부터 시작하자. 숫자 두 개를 곱하는 간단한 함수 $$f(x,y) = x y$$를 생각해보자. 두 입력 각각에 대한 편미분은 간단한 미적분으로 구할 수 있다.

> **Interpretation**. Keep in mind what the derivatives tell you: They indicate the rate of change of a function with respect to that variable surrounding an infinitesimally small region near a particular point:
>
> $$
> \frac{df(x)}{dx} = \lim_{h\ \to 0} \frac{f(x + h) - f(x)}{h}
> $$

**해석.** 도함수가 무엇을 말해주는지 기억해두자. 도함수는 어떤 지점 근처의 무한히 작은 영역에서 그 변수에 대한 함수의 변화율을 나타낸다.

> A technical note is that the division sign on the left-hand side is, unlike the division sign on the right-hand side, not a division. Instead, this notation indicates that the operator $$\frac{d}{dx}$$ is being applied to the function $$f$$, and returns a different function (the derivative). A nice way to think about the expression above is that when $$h$$ is very small, then the function is well-approximated by a straight line, and the derivative is its slope. In other words, the derivative on each variable tells you the sensitivity of the whole expression on its value. For example, if $$x = 4, y = -3$$ then $$f(x,y) = -12$$ and the derivative on $$x$$ $$\frac{\partial f}{\partial x} = -3$$. This tells us that if we were to increase the value of this variable by a tiny amount, the effect on the whole expression would be to decrease it (due to the negative sign), and by three times that amount. This can be seen by rearranging the above equation ( $$f(x + h) = f(x) + h \frac{df(x)}{dx}$$ ). Analogously, since $$\frac{\partial f}{\partial y} = 4$$, we expect that increasing the value of $$y$$ by some very small amount $$h$$ would also increase the output of the function (due to the positive sign), and by $$4h$$.

기술적인 이야기를 덧붙이면, 좌변의 나눗셈 기호는 우변의 나눗셈 기호와 달리 나눗셈이 아니다. 이 표기는 연산자 $$\frac{d}{dx}$$를 함수 $$f$$에 적용해 또 다른 함수(도함수)를 얻는다는 뜻이다. 위 식은 이렇게 생각하면 좋다. $$h$$가 아주 작으면 함수는 직선으로 잘 근사되고, 도함수는 그 직선의 기울기다. 다시 말해 각 변수에 대한 도함수는 그 변수의 값에 식 전체가 얼마나 민감한지를 알려준다. 예를 들어 $$x = 4, y = -3$$이면 $$f(x,y) = -12$$이고 $$x$$에 대한 도함수는 $$\frac{\partial f}{\partial x} = -3$$이다. 이는 이 변수의 값을 아주 조금 키우면 식 전체는 (음의 부호 때문에) 줄어들되 그 세 배만큼 줄어든다는 뜻이다. 위 식을 $$f(x + h) = f(x) + h \frac{df(x)}{dx}$$로 정리해보면 알 수 있다. 마찬가지로 $$\frac{\partial f}{\partial y} = 4$$이므로 $$y$$를 아주 작은 양 $$h$$만큼 키우면 함수의 출력도 (양의 부호 때문에) 커지되 $$4h$$만큼 커진다고 예상할 수 있다.

>> The derivative on each variable tells you the sensitivity of the whole expression on its value.
>
> 각 변수에 대한 도함수는 그 변수의 값에 식 전체가 얼마나 민감한지를 알려준다.

> As mentioned, the gradient $$\nabla f$$ is the vector of partial derivatives, so we have that $$\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]$$. Even though the gradient is technically a vector, we will often use terms such as *“the gradient on x”* instead of the technically correct phrase *“the partial derivative on x”* for simplicity.

앞서 말했듯 기울기 $$\nabla f$$는 편미분을 모아놓은 벡터이므로 $$\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]$$이다. 기울기는 엄밀히 말해 벡터지만, 간단히 쓰기 위해 정확한 표현인 *"x에 대한 편미분"* 대신 *"x에 대한 기울기"* 같은 말을 자주 쓸 것이다.

> We can also derive the derivatives for the addition operation:
>
> $$
> f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial f}{\partial y} = 1
> $$

덧셈 연산에 대한 도함수도 구할 수 있다.

> that is, the derivative on both $$x,y$$ is one regardless of what the values of $$x,y$$ are. This makes sense, since increasing either $$x,y$$ would increase the output of $$f$$, and the rate of that increase would be independent of what the actual values of $$x,y$$ are (unlike the case of multiplication above). The last function we’ll use quite a bit in the class is the *max* operation:
>
> $$
> f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.5in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x)
> $$

즉 $$x,y$$의 값이 무엇이든 두 변수에 대한 도함수는 모두 1이다. 이는 말이 된다. $$x,y$$ 중 어느 쪽을 키워도 $$f$$의 출력은 커지고, 그 증가율은 (위의 곱셈과 달리) $$x,y$$의 실제 값과 무관하기 때문이다. 이 수업에서 꽤 자주 쓸 마지막 함수는 *max* 연산이다.

> That is, the (sub)gradient is 1 on the input that was larger and 0 on the other input. Intuitively, if the inputs are $$x = 4,y = 2$$, then the max is 4, and the function is not sensitive to the setting of $$y$$. That is, if we were to increase it by a tiny amount $$h$$, the function would keep outputting 4, and therefore the gradient is zero: there is no effect. Of course, if we were to change $$y$$ by a large amount (e.g. larger than 2), then the value of $$f$$ would change, but the derivatives tell us nothing about the effect of such large changes on the inputs of a function; They are only informative for tiny, infinitesimally small changes on the inputs, as indicated by the $$\lim_{h \rightarrow 0}$$ in its definition.

즉 (부분)기울기는 더 컸던 입력 쪽에서 1이고 다른 입력 쪽에서는 0이다. 직관적으로 보면, 입력이 $$x = 4,y = 2$$일 때 최댓값은 4이고 이 함수는 $$y$$를 어떻게 두든 민감하지 않다. 즉 $$y$$를 아주 작은 양 $$h$$만큼 키워도 함수는 계속 4를 내놓으므로 기울기는 0이다. 아무 영향이 없는 것이다. 물론 $$y$$를 크게 (예컨대 2보다 크게) 바꾸면 $$f$$의 값도 달라지겠지만, 도함수는 입력을 그렇게 크게 바꿨을 때의 영향에 대해서는 아무것도 말해주지 않는다. 정의에 들어 있는 $$\lim_{h \rightarrow 0}$$이 나타내듯, 도함수는 입력에 준 아주 작은, 무한히 작은 변화에 대해서만 유효한 정보다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** $$\mathbb{1}(\cdot)$$는 괄호 안의 조건이 참이면 1, 거짓이면 0을 내놓는 지시 함수다.
> 두 입력이 정확히 같을 때($$x = y$$)에는 위 식이 양쪽 모두에 1을 주는데, 바로 그 지점이 max
> 함수의 꺾임이라 도함수가 정의되지 않는다. 이때 위 식이 내놓는 $$(1,1)$$은 사실 부분기울기가
> 아니다 — 진짜 부분기울기는 두 성분의 합이 1이 되는 조합($$(1,0)$$, $$(0.5,0.5)$$ 등)이며,
> 실제 구현은 보통 한쪽에만 1을 주어 아래로 흘려보내는 기울기의 총합이 1이 되도록 한다.
{: .prompt-tip }
<!-- markdownlint-restore -->

<a id="backprop"></a>

### Compound expressions with chain rule

> Lets now start to consider more complicated expressions that involve multiple composed functions, such as $$f(x,y,z) = (x + y) z$$. This expression is still simple enough to differentiate directly, but we’ll take a particular approach to it that will be helpful with understanding the intuition behind backpropagation. In particular, note that this expression can be broken down into two expressions: $$q = x + y$$ and $$f = q z$$. Moreover, we know how to compute the derivatives of both expressions separately, as seen in the previous section. $$f$$ is just multiplication of $$q$$ and $$z$$, so $$\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q$$, and $$q$$ is addition of $$x$$ and $$y$$ so $$\frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1$$. However, we don’t necessarily care about the gradient on the intermediate value $$q$$ - the value of $$\frac{\partial f}{\partial q}$$ is not useful. Instead, we are ultimately interested in the gradient of $$f$$ with respect to its inputs $$x,y,z$$. The **chain rule** tells us that the correct way to “chain” these gradient expressions together is through multiplication. For example, $$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x}$$. In practice this is simply a multiplication of the two numbers that hold the two gradients. Lets see this with an example:

이제 여러 함수가 합성된 좀 더 복잡한 식, 예컨대 $$f(x,y,z) = (x + y) z$$를 생각해보자. 이 식도 곧바로 미분할 수 있을 만큼 단순하지만, 역전파의 직관을 이해하는 데 도움이 되는 특별한 방식으로 접근해보겠다. 특히 이 식이 $$q = x + y$$와 $$f = q z$$라는 두 식으로 쪼개진다는 데 주목하자. 게다가 앞 절에서 봤듯 두 식 각각의 도함수는 어떻게 구하는지 이미 안다. $$f$$는 $$q$$와 $$z$$의 곱일 뿐이므로 $$\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q$$이고, $$q$$는 $$x$$와 $$y$$의 합이므로 $$\frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1$$이다. 그런데 우리는 중간 값 $$q$$에 대한 기울기 자체에는 딱히 관심이 없다. $$\frac{\partial f}{\partial q}$$ 값은 그 자체로는 쓸모가 없다. 우리가 끝내 알고 싶은 것은 입력 $$x,y,z$$에 대한 $$f$$의 기울기다. **연쇄 법칙**은 이 기울기 식들을 서로 "이어붙이는" 올바른 방법이 곱셈임을 알려준다. 예를 들어 $$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x}$$이다. 실제로 하는 일은 두 기울기를 담은 두 숫자를 곱하는 것에 지나지 않는다. 예제로 살펴보자.

```python
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
dqdx = 1.0
dqdy = 1.0
# now backprop through q = x + y
dfdx = dfdq * dqdx  # The multiplication here is the chain rule!
dfdy = dfdq * dqdy  
```

> We are left with the gradient in the variables `[dfdx,dfdy,dfdz]`, which tell us the sensitivity of the variables `x,y,z` on `f`!. This is the simplest example of backpropagation. Going forward, we will use a more concise notation that omits the `df` prefix. For example, we will simply write `dq` instead of `dfdq`, and always assume that the gradient is computed on the final output.

이렇게 해서 변수 `[dfdx,dfdy,dfdz]`에 기울기가 남는데, 이 값들은 변수 `x,y,z`에 대해 `f`가 얼마나 민감한지를 알려준다! 이것이 역전파의 가장 간단한 예다. 앞으로는 `df` 접두사를 뗀 더 간결한 표기를 쓰겠다. 예컨대 `dfdq` 대신 그냥 `dq`라고 쓰고, 기울기는 언제나 최종 출력에 대해 계산한 것으로 본다.

> This computation can also be nicely visualized with a circuit diagram:

이 계산은 회로 그림으로도 보기 좋게 나타낼 수 있다.

<div style="background:#fff; padding:1rem; border-radius:8px; overflow-x:auto">
<svg style="max-width: 420px" viewbox="0 0 420 220"><defs><marker id="arrowhead" refx="6" refy="2" markerwidth="6" markerheight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="40" y1="30" x2="110" y2="30" stroke="black" stroke-width="1"></line><text x="45" y="24" font-size="16" fill="green">-2</text><text x="45" y="47" font-size="16" fill="red">-4</text><text x="35" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="40" y1="100" x2="110" y2="100" stroke="black" stroke-width="1"></line><text x="45" y="94" font-size="16" fill="green">5</text><text x="45" y="117" font-size="16" fill="red">-4</text><text x="35" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="40" y1="170" x2="110" y2="170" stroke="black" stroke-width="1"></line><text x="45" y="164" font-size="16" fill="green">-4</text><text x="45" y="187" font-size="16" fill="red">3</text><text x="35" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="210" y1="65" x2="280" y2="65" stroke="black" stroke-width="1"></line><text x="215" y="59" font-size="16" fill="green">3</text><text x="215" y="82" font-size="16" fill="red">-4</text><text x="205" y="59" font-size="16" text-anchor="end" fill="black">q</text><circle cx="170" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="170" y="70" font-size="20" fill="black" text-anchor="middle">+</text><line x1="110" y1="30" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="100" x2="150" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="190" y1="65" x2="210" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="380" y1="117" x2="450" y2="117" stroke="black" stroke-width="1"></line><text x="385" y="111" font-size="16" fill="green">-12</text><text x="385" y="134" font-size="16" fill="red">1</text><text x="375" y="111" font-size="16" text-anchor="end" fill="black">f</text><circle cx="340" cy="117" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="340" y="127" font-size="20" fill="black" text-anchor="middle">*</text><line x1="280" y1="65" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="110" y1="170" x2="320" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="360" y1="117" x2="380" y2="117" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></svg>
</div>

_The real-valued *"circuit"* on left shows the visual representation of the computation. The **forward pass** computes values from inputs to output (shown in green). The **backward pass** then performs backpropagation which starts at the end and recursively applies the chain rule to compute the gradients (shown in red) all the way to the inputs of the circuit. The gradients can be thought of as flowing backwards through the circuit._

왼쪽의 실숫값 *"회로"*는 계산 과정을 그림으로 나타낸 것이다. **순전파**는 입력에서 출력 쪽으로 값을 계산한다(초록색). 이어지는 **역전파**는 끝에서 시작해 연쇄 법칙을 재귀적으로 적용하며 회로의 입력까지 기울기를 계산해 내려간다(빨간색). 기울기가 회로를 거꾸로 흘러간다고 생각하면 된다.

<a id="intuitive"></a>

### Intuitive understanding of backpropagation

> Notice that backpropagation is a beautifully local process. Every gate in a circuit diagram gets some inputs and can right away compute two things: 1. its output value and 2. the *local* gradient of its output with respect to its inputs. Notice that the gates can do this completely independently without being aware of any of the details of the full circuit that they are embedded in. However, once the forward pass is over, during backpropagation the gate will eventually learn about the gradient of its output value on the final output of the entire circuit. Chain rule says that the gate should take that gradient and multiply it into every gradient it normally computes for all of its inputs.

역전파가 아름다우리만치 국소적인 과정이라는 점에 주목하자. 회로 그림의 모든 게이트는 입력 몇 개를 받으면 곧바로 두 가지를 계산할 수 있다. 1. 자기 출력 값, 2. 자기 입력에 대한 자기 출력의 *국소 기울기(local gradient)*다. 게이트가 자신이 속한 전체 회로의 사정을 전혀 모르는 채 완전히 독립적으로 이 일을 할 수 있다는 점에 주목하자. 다만 순전파가 끝나고 역전파가 진행되면 게이트는 결국 전체 회로의 최종 출력에 대한 자기 출력 값의 기울기를 알게 된다. 연쇄 법칙에 따르면 게이트는 그 기울기를 받아, 자기 입력들에 대해 원래 계산하던 기울기 각각에 곱해주면 된다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** 이 문단이 역전파의 핵심이다. 게이트가 입력 $$a$$를 받아 출력 $$z$$를 내놓고
> 회로 전체의 최종 출력이 $$L$$이라고 하자. 게이트가 다루는 기울기는 두 종류다.
> 하나는 **국소 기울기** $$\frac{\partial z}{\partial a}$$로, 순전파 때 받은 입력만 있으면
> 회로의 나머지를 전혀 몰라도 스스로 구할 수 있다. 다른 하나는 역전파 때 위에서 내려오는
> **상류 기울기**(upstream gradient) $$\frac{\partial L}{\partial z}$$로, 게이트 바깥이
> 계산해 건네주는 값이다. 게이트가 할 일은 이 둘을 곱해
> $$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial a}$$를
> 만들어 아래로 내려보내는 것뿐이다. 국소 기울기는 회로를 몰라도 되고 상류 기울기는 게이트
> 속을 몰라도 되므로, 이 분업 덕분에 어떤 게이트든 마음대로 갈아 끼울 수 있다. '상류'라는
> 말은 원문에 없지만 이후 문헌에서 널리 쓰인다.
{: .prompt-tip }
<!-- markdownlint-restore -->

>> This extra multiplication (for each input) due to the chain rule can turn a single and relatively useless gate into a cog in a complex circuit such as an entire neural network.
>
> 연쇄 법칙 때문에 (입력마다) 한 번씩 더 하는 이 곱셈이, 홀로 있을 때는 별 쓸모없던 게이트 하나를 신경망 전체 같은 복잡한 회로의 톱니바퀴로 만들어준다.

> Lets get an intuition for how this works by referring again to the example. The add gate received inputs [-2, 5] and computed output 3. Since the gate is computing the addition operation, its local gradient for both of its inputs is +1. The rest of the circuit computed the final value, which is -12. During the backward pass in which the chain rule is applied recursively backwards through the circuit, the add gate (which is an input to the multiply gate) learns that the gradient for its output was -4. If we anthropomorphize the circuit as wanting to output a higher value (which can help with intuition), then we can think of the circuit as “wanting” the output of the add gate to be lower (due to negative sign), and with a *force* of 4. To continue the recurrence and to chain the gradient, the add gate takes that gradient and multiplies it to all of the local gradients for its inputs (making the gradient on both **x** and **y** 1 * -4 = -4). Notice that this has the desired effect: If **x,y** were to decrease (responding to their negative gradient) then the add gate’s output would decrease, which in turn makes the multiply gate’s output increase.

이것이 어떻게 돌아가는지 앞의 예제를 다시 보며 직관을 얻어보자. 덧셈 게이트는 입력 [-2, 5]를 받아 출력 3을 계산했다. 이 게이트는 덧셈 연산을 하므로 두 입력에 대한 국소 기울기는 모두 +1이다. 회로의 나머지 부분은 최종 값 -12를 계산했다. 회로를 거슬러 올라가며 연쇄 법칙을 재귀적으로 적용하는 역전파 과정에서, (곱셈 게이트의 입력인) 덧셈 게이트는 자기 출력에 대한 기울기가 -4임을 알게 된다. 회로를 의인화해서 더 큰 값을 출력하고 싶어 한다고 보면(직관에 도움이 된다), 회로가 덧셈 게이트의 출력이 (음의 부호 때문에) 더 낮아지기를 4만큼의 *힘*으로 "바란다"고 생각할 수 있다. 재귀를 이어가고 기울기를 연결하기 위해 덧셈 게이트는 그 기울기를 받아 자기 입력들의 국소 기울기 모두에 곱한다(그래서 **x**와 **y**에 대한 기울기가 모두 1 * -4 = -4가 된다). 이것이 바라던 효과를 낸다는 점에 주목하자. **x,y**가 (자기 기울기가 음수인 데 따라) 줄어들면 덧셈 게이트의 출력이 줄어들고, 그러면 곱셈 게이트의 출력은 커진다.

> Backpropagation can thus be thought of as gates communicating to each other (through the gradient signal) whether they want their outputs to increase or decrease (and how strongly), so as to make the final output value higher.

따라서 역전파는 게이트들이 (기울기 신호를 통해) 서로에게 자기 출력이 커지기를 원하는지 작아지기를 원하는지, 그리고 얼마나 강하게 원하는지를 전달해 최종 출력 값을 더 높이려 하는 과정이라고 볼 수 있다.

<a id="sigmoid"></a>

### Modularity: Sigmoid example

> The gates we introduced above are relatively arbitrary. Any kind of differentiable function can act as a gate, and we can group multiple gates into a single gate, or decompose a function into multiple gates whenever it is convenient. Lets look at another expression that illustrates this point:
>
> $$
> f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}
> $$

위에서 소개한 게이트들은 다분히 임의적이다. 미분 가능한 함수라면 무엇이든 게이트가 될 수 있고, 편할 때마다 여러 게이트를 하나로 묶거나 함수 하나를 여러 게이트로 쪼갤 수 있다. 이 점을 보여주는 다른 식을 보자.

> as we will see later in the class, this expression describes a 2-dimensional neuron (with inputs **x** and weights **w**) that uses the *sigmoid activation* function. But for now lets think of this very simply as just a function from inputs *w,x* to a single number. The function is made up of multiple gates. In addition to the ones described already above (add, mul, max), there are four more:
>
> $$
> f(x) = \frac{1}{x} 
> \hspace{1in} \rightarrow \hspace{1in} 
> \frac{df}{dx} = -1/x^2 
> \\\\
> f_c(x) = c + x
> \hspace{1in} \rightarrow \hspace{1in} 
> \frac{df}{dx} = 1 
> \\\\
> f(x) = e^x
> \hspace{1in} \rightarrow \hspace{1in} 
> \frac{df}{dx} = e^x
> \\\\
> f_a(x) = ax
> \hspace{1in} \rightarrow \hspace{1in} 
> \frac{df}{dx} = a
> $$

수업 뒤쪽에서 보겠지만 이 식은 *sigmoid 활성화* 함수를 쓰는 2차원 뉴런(입력은 **x**, 가중치는 **w**)을 나타낸다. 하지만 지금은 그저 입력 *w,x*를 숫자 하나로 보내는 함수로만 생각하자. 이 함수는 여러 게이트로 이루어져 있다. 이미 위에서 설명한 것들(add, mul, max) 말고도 네 개가 더 있다.

> Where the functions $$f_c, f_a$$ translate the input by a constant of $$c$$ and scale the input by a constant of $$a$$, respectively. These are technically special cases of addition and multiplication, but we introduce them as (new) unary gates here since we do not need the gradients for the constants $$c,a$$. The full circuit then looks as follows:

여기서 함수 $$f_c, f_a$$는 각각 입력을 상수 $$c$$만큼 평행이동하고 상수 $$a$$배로 늘린다. 엄밀히 말하면 덧셈과 곱셈의 특수한 경우지만, 상수 $$c,a$$에 대한 기울기는 필요 없으므로 여기서는 (새로운) 단항 게이트로 소개한다. 그러면 전체 회로는 다음과 같은 모습이 된다.

<div style="background:#fff; padding:1rem; border-radius:8px; overflow-x:auto">
<svg style="max-width: 799px" viewbox="0 0 799 306"><g transform="scale(0.8)"><defs><marker id="arrowhead" refx="6" refy="2" markerwidth="6" markerheight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">2.00</text><text x="55" y="47" font-size="16" fill="red">-0.20</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">w0</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-1.00</text><text x="55" y="117" font-size="16" fill="red">0.39</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">x0</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">-3.00</text><text x="55" y="187" font-size="16" fill="red">-0.39</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">w1</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-2.00</text><text x="55" y="257" font-size="16" fill="red">-0.59</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">x1</text><line x1="50" y1="310" x2="90" y2="310" stroke="black" stroke-width="1"></line><text x="55" y="304" font-size="16" fill="green">-3.00</text><text x="55" y="327" font-size="16" fill="red">0.20</text><text x="45" y="304" font-size="16" text-anchor="end" fill="black">w2</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-2.00</text><text x="175" y="82" font-size="16" fill="red">0.20</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">6.00</text><text x="175" y="222" font-size="16" fill="red">0.20</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="215" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">4.00</text><text x="295" y="152" font-size="16" fill="red">0.20</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="222" x2="450" y2="222" stroke="black" stroke-width="1"></line><text x="415" y="216" font-size="16" fill="green">1.00</text><text x="415" y="239" font-size="16" fill="red">0.20</text><circle cx="370" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="227" font-size="20" fill="black" text-anchor="middle">+</text><line x1="330" y1="135" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="310" x2="350" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="222" x2="410" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="530" y1="222" x2="570" y2="222" stroke="black" stroke-width="1"></line><text x="535" y="216" font-size="16" fill="green">-1.00</text><text x="535" y="239" font-size="16" fill="red">-0.20</text><circle cx="490" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="490" y="227" font-size="20" fill="black" text-anchor="middle">*-1</text><line x1="450" y1="222" x2="470" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="510" y1="222" x2="530" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="650" y1="222" x2="690" y2="222" stroke="black" stroke-width="1"></line><text x="655" y="216" font-size="16" fill="green">0.37</text><text x="655" y="239" font-size="16" fill="red">-0.53</text><circle cx="610" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="610" y="227" font-size="20" fill="black" text-anchor="middle">exp</text><line x1="570" y1="222" x2="590" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="630" y1="222" x2="650" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="770" y1="222" x2="810" y2="222" stroke="black" stroke-width="1"></line><text x="775" y="216" font-size="16" fill="green">1.37</text><text x="775" y="239" font-size="16" fill="red">-0.53</text><circle cx="730" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="730" y="227" font-size="20" fill="black" text-anchor="middle">+1</text><line x1="690" y1="222" x2="710" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="750" y1="222" x2="770" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="890" y1="222" x2="930" y2="222" stroke="black" stroke-width="1"></line><text x="895" y="216" font-size="16" fill="green">0.73</text><text x="895" y="239" font-size="16" fill="red">1.00</text><circle cx="850" cy="222" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="850" y="227" font-size="20" fill="black" text-anchor="middle">1/x</text><line x1="810" y1="222" x2="830" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="870" y1="222" x2="890" y2="222" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
</div>

_Example circuit for a 2D neuron with a sigmoid activation function. The inputs are [x0,x1] and the (learnable) weights of the neuron are [w0,w1,w2]. As we will see later, the neuron computes a dot product with the input and then its activation is softly squashed by the sigmoid function to be in range from 0 to 1._

sigmoid 활성화 함수를 쓰는 2D 뉴런의 예시 회로. 입력은 [x0,x1]이고 뉴런의 (학습되는) 가중치는 [w0,w1,w2]다. 뒤에서 보겠지만 이 뉴런은 입력과의 내적을 계산한 다음, 그 활성값을 sigmoid 함수로 0에서 1 사이에 부드럽게 눌러 넣는다.

> In the example above, we see a long chain of function applications that operates on the result of the dot product between **w,x**. The function that these operations implement is called the *sigmoid function* $$\sigma(x)$$. It turns out that the derivative of the sigmoid function with respect to its input simplifies if you perform the derivation (after a fun tricky part where we add and subtract a 1 in the numerator):
>
> $$
> \sigma(x) = \frac{1}{1+e^{-x}} \\\\
> \rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) 
> = \left( 1 - \sigma(x) \right) \sigma(x)
> $$

위 예제에서는 **w,x**의 내적 결과에 함수들이 길게 이어져 적용되는 것을 볼 수 있다. 이 연산들이 구현하는 함수를 *sigmoid 함수* $$\sigma(x)$$라고 부른다. 그런데 sigmoid 함수를 그 입력에 대해 미분해보면 (분자에 1을 더했다 빼는 재미있는 요령을 거쳐) 식이 간단해진다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** '분자에 1을 더했다 빼는 재미있는 요령'이란 $$e^{-x} = (1 + e^{-x}) - 1$$로 고쳐
> 쓰는 것이다. 그러면 분자와 분모가 나란히 정리된다.
>
> $$
> \frac{e^{-x}}{(1+e^{-x})^2} = \frac{(1 + e^{-x}) - 1}{1+e^{-x}} \cdot \frac{1}{1+e^{-x}} = \left( 1 - \sigma(x) \right) \sigma(x)
> $$
>
> 즉 sigmoid의 도함수는 순전파에서 이미 계산해둔 출력값 $$\sigma(x)$$ 하나만 있으면 구할 수
> 있다. 입력 $$x$$를 다시 꺼내 지수 함수를 계산할 필요가 없다는 뜻이고, 이것이 sigmoid를
> 게이트 하나로 묶어두는 실질적인 이득이다.
{: .prompt-tip }
<!-- markdownlint-restore -->

> As we see, the gradient turns out to simplify and becomes surprisingly simple. For example, the sigmoid expression receives the input 1.0 and computes the output 0.73 during the forward pass. The derivation above shows that the *local* gradient would simply be (1 - 0.73) * 0.73 ~= 0.2, as the circuit computed before (see the image above), except this way it would be done with a single, simple and efficient expression (and with less numerical issues). Therefore, in any real practical application it would be very useful to group these operations into a single gate. Lets see the backprop for this neuron in code:

보다시피 기울기는 놀랍도록 간단하게 정리된다. 예를 들어 순전파에서 sigmoid 식은 입력 1.0을 받아 출력 0.73을 계산한다. 위 유도에 따르면 *국소* 기울기는 그저 (1 - 0.73) * 0.73 ~= 0.2이며, 이는 회로가 앞서 계산한 값(위 그림 참고)과 같다. 다만 이렇게 하면 간단하고 효율적인 식 하나로 끝나고 수치적 문제도 덜하다. 그러므로 실제 응용에서는 이 연산들을 게이트 하나로 묶어두는 것이 아주 유용하다. 이 뉴런의 역전파를 코드로 보자.

```python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```

> **Implementation protip: staged backpropagation**. As shown in the code above, in practice it is always helpful to break down the forward pass into stages that are easily backpropped through. For example here we created an intermediate variable `dot` which holds the output of the dot product between `w` and `x`. During backward pass we then successively compute (in reverse order) the corresponding variables (e.g. `ddot`, and ultimately `dw, dx`) that hold the gradients of those variables.

**구현 요령: 단계별 역전파.** 위 코드에서 보듯 실전에서는 순전파를 역전파하기 쉬운 단계들로 쪼개는 것이 언제나 도움이 된다. 예를 들어 여기서는 `w`와 `x`의 내적 결과를 담는 중간 변수 `dot`을 만들었다. 그다음 역전파에서는 그 변수들의 기울기를 담는 대응 변수(예컨대 `ddot`, 최종적으로는 `dw, dx`)를 역순으로 차례차례 계산한다.

> The point of this section is that the details of how the backpropagation is performed, and which parts of the forward function we think of as gates, is a matter of convenience. It helps to be aware of which parts of the expression have easy local gradients, so that they can be chained together with the least amount of code and effort.

이 절의 요점은, 역전파를 구체적으로 어떻게 수행할지 그리고 순전파 함수의 어느 부분을 게이트로 볼지는 편의의 문제라는 것이다. 식의 어느 부분이 국소 기울기를 쉽게 구할 수 있는 부분인지 알아두면 도움이 된다. 그래야 가장 적은 코드와 수고로 그것들을 연쇄시킬 수 있다.

<a id="staged"></a>

### Backprop in practice: Staged computation

> Lets see this with another example. Suppose that we have a function of the form:
>
> $$
> f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}
> $$

다른 예제로 살펴보자. 다음과 같은 형태의 함수가 있다고 하자.

> To be clear, this function is completely useless and it’s not clear why you would ever want to compute its gradient, except for the fact that it is a good example of backpropagation in practice. It is very important to stress that if you were to launch into performing the differentiation with respect to either $$x$$ or $$y$$, you would end up with very large and complex expressions. However, it turns out that doing so is completely unnecessary because we don’t need to have an explicit function written down that evaluates the gradient. We only have to know how to compute it. Here is how we would structure the forward pass of such expression:

분명히 해두자면 이 함수는 전혀 쓸모가 없고 그 기울기를 계산할 이유도 딱히 없다. 실전 역전파의 좋은 예라는 점만이 유일한 이유다. 꼭 강조하고 싶은 것은, $$x$$나 $$y$$에 대해 곧바로 미분에 뛰어들면 아주 크고 복잡한 식이 나오고 만다는 점이다. 그런데 그렇게 할 필요가 전혀 없다. 기울기를 계산하는 명시적인 함수를 적어둘 필요가 없기 때문이다. 우리는 그것을 어떻게 계산하는지만 알면 된다. 이런 식의 순전파는 다음과 같이 짜면 된다.

```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```

> Phew, by the end of the expression we have computed the forward pass. Notice that we have structured the code in such way that it contains multiple intermediate variables, each of which are only simple expressions for which we already know the local gradients. Therefore, computing the backprop pass is easy: We’ll go backwards and for every variable along the way in the forward pass (`sigy, num, sigx, xpy, xpysqr, den, invden`) we will have the same variable, but one that begins with a `d`, which will hold the gradient of the output of the circuit with respect to that variable. Additionally, note that every single piece in our backprop will involve computing the local gradient of that expression, and chaining it with the gradient on that expression with a multiplication. For each row, we also highlight which part of the forward pass it refers to:

휴, 식 끝까지 와서 순전파를 계산했다. 코드를 중간 변수를 여럿 두는 방식으로 짰다는 점에 주목하자. 각 중간 변수는 우리가 이미 국소 기울기를 알고 있는 간단한 식으로만 이루어져 있다. 그래서 역전파를 계산하기가 쉽다. 거꾸로 올라가면서 순전파에 나온 변수(`sigy, num, sigx, xpy, xpysqr, den, invden`)마다 이름 앞에 `d`가 붙은 같은 변수를 두고, 거기에 그 변수에 대한 회로 출력의 기울기를 담는다. 또한 역전파의 모든 조각이 그 식의 국소 기울기를 계산한 뒤 그 식에 대한 기울기와 곱셈으로 이어붙이는 일로 이루어진다는 점에 유의하자. 각 줄에는 그것이 순전파의 어느 부분에 해당하는지도 함께 표시해두었다.

```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

> Notice a few things:

몇 가지 짚어둘 것이 있다.

> **Cache forward pass variables**. To compute the backward pass it is very helpful to have some of the variables that were used in the forward pass. In practice you want to structure your code so that you cache these variables, and so that they are available during backpropagation. If this is too difficult, it is possible (but wasteful) to recompute them.

**순전파 변수를 캐시해둘 것**. 역전파를 계산하려면 순전파에서 썼던 변수 중 일부가 있어야 아주 편하다. 실전에서는 이 변수들을 캐시해두어 역전파 때 바로 쓸 수 있도록 코드를 짜는 것이 좋다. 그렇게 하기가 너무 어렵다면 (낭비이긴 하지만) 다시 계산해도 된다.

> **Gradients add up at forks**. The forward expression involves the variables **x,y** multiple times, so when we perform backpropagation we must be careful to use `+=` instead of `=` to accumulate the gradient on these variables (otherwise we would overwrite it). This follows the *multivariable chain rule* in Calculus, which states that if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add.

**갈림길에서는 기울기가 더해진다**. 순전파 식에 변수 **x,y**가 여러 번 나오므로, 역전파할 때는 이 변수들의 기울기를 `=`가 아니라 `+=`로 누적하도록 조심해야 한다(그러지 않으면 앞서 쌓아둔 값을 덮어쓴다). 이는 미적분의 *다변수 연쇄 법칙*을 따르는 것으로, 한 변수가 회로의 여러 갈래로 뻗어나가면 그 변수로 되돌아오는 기울기들은 더해진다는 법칙이다.

### 보충: 단계별 계산을 실제로 검산해보기

원문의 순전파와 역전파 코드를 그대로 함수로 옮긴 다음, 03번에서 본 중심 차분 공식으로 구한
수치적 기울기와 비교해보자. 덤으로 `+=`를 `=`로 바꾸면 어떤 값이 나오는지도 함께 찍어본다.

```python
import math

def f(x, y):                                  # 원문의 순전파 (1)~(8)
    sigy = 1.0 / (1 + math.exp(-y))
    num = x + sigy
    sigx = 1.0 / (1 + math.exp(-x))
    xpy = x + y
    xpysqr = xpy**2
    den = sigx + xpysqr
    invden = 1.0 / den
    return num * invden

def grad(x, y, accumulate=True):              # 원문의 역전파 (8)~(1)
    sigy = 1.0 / (1 + math.exp(-y))
    num = x + sigy
    sigx = 1.0 / (1 + math.exp(-x))
    xpy = x + y
    xpysqr = xpy**2
    den = sigx + xpysqr
    invden = 1.0 / den

    dnum, dinvden = invden, num
    dden = (-1.0 / (den**2)) * dinvden
    dsigx = (1) * dden
    dxpysqr = (1) * dden
    dxpy = (2 * xpy) * dxpysqr
    dx = (1) * dxpy
    dy = (1) * dxpy
    if accumulate:                            # 원문 그대로: +=
        dx += ((1 - sigx) * sigx) * dsigx
        dx += (1) * dnum
        dsigy = (1) * dnum
        dy += ((1 - sigy) * sigy) * dsigy
    else:                                     # += 를 = 로 바꿔본 경우
        dx = ((1 - sigx) * sigx) * dsigx
        dx = (1) * dnum
        dsigy = (1) * dnum
        dy = ((1 - sigy) * sigy) * dsigy
    return dx, dy

x, y, h = 3.0, -4.0, 1e-5
dx, dy = grad(x, y)
print("해석적    dx = %.10f  dy = %.10f" % (dx, dy))
print("수치적    dx = %.10f  dy = %.10f" % ((f(x + h, y) - f(x - h, y)) / (2 * h),
                                            (f(x, y + h) - f(x, y - h)) / (2 * h)))
print("= 로 쓰면 dx = %.10f  dy = %.10f" % grad(x, y, accumulate=False))
```

```text
해석적    dx = 2.0595697956  dy = 1.5922327515
수치적    dx = 2.0595697956  dy = 1.5922327515
= 로 쓰면 dx = 0.5121444488  dy = 0.0090458569
```

해석적 기울기와 수치적 기울기가 소수점 열째 자리까지 맞는다. 반면 `+=`를 `=`로 바꾸면 $$x$$는
갈림길 세 곳(`num`, `sigx`, `xpy`) 가운데 마지막으로 대입된 하나의 기여만, $$y$$는 두
곳(`sigy`, `xpy`) 가운데 하나의 기여만 남아 완전히 다른 값이 된다. 바로 위 문단의 경고가
문장이 아니라 숫자로 확인되는 셈이다.

<a id="patterns"></a>

### Patterns in backward flow

> It is interesting to note that in many cases the backward-flowing gradient can be interpreted on an intuitive level. For example, the three most commonly used gates in neural networks (*add,mul,max*), all have very simple interpretations in terms of how they act during backpropagation. Consider this example circuit:

역방향으로 흘러오는 기울기를 직관적인 수준에서 해석할 수 있는 경우가 많다는 점은 흥미롭다. 예를 들어 신경망에서 가장 많이 쓰는 게이트 셋(*add, mul, max*)은 모두 역전파에서 어떻게 동작하는지가 아주 간단하게 해석된다. 다음 예시 회로를 보자.

<div style="background:#fff; padding:1rem; border-radius:8px; overflow-x:auto">
<svg style="max-width: 460px" viewbox="0 0 460 290"><g transform="scale(1)"><defs><marker id="arrowhead" refx="6" refy="2" markerwidth="6" markerheight="4" orient="auto"><path d="M 0,0 V 4 L6,2 Z"></path></marker></defs><line x1="50" y1="30" x2="90" y2="30" stroke="black" stroke-width="1"></line><text x="55" y="24" font-size="16" fill="green">3.00</text><text x="55" y="47" font-size="16" fill="red">-8.00</text><text x="45" y="24" font-size="16" text-anchor="end" fill="black">x</text><line x1="50" y1="100" x2="90" y2="100" stroke="black" stroke-width="1"></line><text x="55" y="94" font-size="16" fill="green">-4.00</text><text x="55" y="117" font-size="16" fill="red">6.00</text><text x="45" y="94" font-size="16" text-anchor="end" fill="black">y</text><line x1="50" y1="170" x2="90" y2="170" stroke="black" stroke-width="1"></line><text x="55" y="164" font-size="16" fill="green">2.00</text><text x="55" y="187" font-size="16" fill="red">2.00</text><text x="45" y="164" font-size="16" text-anchor="end" fill="black">z</text><line x1="50" y1="240" x2="90" y2="240" stroke="black" stroke-width="1"></line><text x="55" y="234" font-size="16" fill="green">-1.00</text><text x="55" y="257" font-size="16" fill="red">0.00</text><text x="45" y="234" font-size="16" text-anchor="end" fill="black">w</text><line x1="170" y1="65" x2="210" y2="65" stroke="black" stroke-width="1"></line><text x="175" y="59" font-size="16" fill="green">-12.00</text><text x="175" y="82" font-size="16" fill="red">2.00</text><circle cx="130" cy="65" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="75" font-size="20" fill="black" text-anchor="middle">*</text><line x1="90" y1="30" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="100" x2="110" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="65" x2="170" y2="65" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="170" y1="205" x2="210" y2="205" stroke="black" stroke-width="1"></line><text x="175" y="199" font-size="16" fill="green">2.00</text><text x="175" y="222" font-size="16" fill="red">2.00</text><circle cx="130" cy="205" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="130" y="210" font-size="20" fill="black" text-anchor="middle">max</text><line x1="90" y1="170" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="90" y1="240" x2="110" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="150" y1="205" x2="170" y2="205" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="290" y1="135" x2="330" y2="135" stroke="black" stroke-width="1"></line><text x="295" y="129" font-size="16" fill="green">-10.00</text><text x="295" y="152" font-size="16" fill="red">2.00</text><circle cx="250" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="250" y="140" font-size="20" fill="black" text-anchor="middle">+</text><line x1="210" y1="65" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="210" y1="205" x2="230" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="270" y1="135" x2="290" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="410" y1="135" x2="450" y2="135" stroke="black" stroke-width="1"></line><text x="415" y="129" font-size="16" fill="green">-20.00</text><text x="415" y="152" font-size="16" fill="red">1.00</text><circle cx="370" cy="135" fill="white" stroke="black" stroke-width="1" r="20"></circle><text x="370" y="140" font-size="20" fill="black" text-anchor="middle">*2</text><line x1="330" y1="135" x2="350" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line><line x1="390" y1="135" x2="410" y2="135" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line></g></svg>
</div>

_An example circuit demonstrating the intuition behind the operations that backpropagation performs during the backward pass in order to compute the gradients on the inputs. Sum operation distributes gradients equally to all its inputs. Max operation routes the gradient to the higher input. Multiply gate takes the input activations, swaps them and multiplies by its gradient._

입력에 대한 기울기를 계산하기 위해 역전파가 역방향 진행 중에 수행하는 연산들, 그 직관을 보여주는 예시 회로. 합 연산은 기울기를 모든 입력에 똑같이 나눠준다. max 연산은 기울기를 더 큰 입력 쪽으로 흘려보낸다. 곱셈 게이트는 입력 활성값을 서로 맞바꾼 다음 자기 기울기와 곱한다.

> Looking at the diagram above as an example, we can see that:

위 그림을 예로 삼아 보면 다음을 알 수 있다.

> The **add gate** always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass. This follows from the fact that the local gradient for the add operation is simply +1.0, so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged). In the example circuit above, note that the + gate routed the gradient of 2.00 to both of its inputs, equally and unchanged.

**덧셈 게이트**는 언제나 자기 출력에 대한 기울기를 받아 모든 입력에 똑같이 나눠준다. 순전파 때 입력 값이 무엇이었든 상관없다. 덧셈 연산의 국소 기울기가 그저 +1.0이기 때문이다. x1.0을 곱하는 셈이라 값이 그대로 남으므로, 모든 입력에 대한 기울기는 출력에 대한 기울기와 정확히 같아진다. 위 예시 회로에서 + 게이트가 기울기 2.00을 두 입력 모두에 똑같이, 값을 바꾸지 않고 흘려보냈다는 점에 주목하자.

> The **max gate** routes the gradient. Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass). This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values. In the example circuit above, the max operation routed the gradient of 2.00 to the **z** variable, which had a higher value than **w**, and the gradient on **w** remains zero.

**max 게이트**는 기울기의 경로를 정한다. 기울기를 값 그대로 모든 입력에 나눠주는 덧셈 게이트와 달리, max 게이트는 기울기를 (값은 그대로 둔 채) 정확히 입력 하나에게만 보낸다. 순전파 때 값이 가장 컸던 입력이다. max 게이트의 국소 기울기가 가장 큰 값에 대해서는 1.0이고 나머지 값에 대해서는 모두 0.0이기 때문이다. 위 예시 회로에서 max 연산은 기울기 2.00을 **w**보다 값이 컸던 **z** 쪽으로 흘려보냈고, **w**에 대한 기울기는 0으로 남았다.

> The **multiply gate** is a little less easy to interpret. Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule. In the example above, the gradient on **x** is -8.00, which is -4.00 x 2.00.

**곱셈 게이트**는 해석하기가 조금 덜 쉽다. 이 게이트의 국소 기울기는 입력 값들인데 서로 맞바뀌어 있고, 연쇄 법칙에 따라 여기에 자기 출력에 대한 기울기가 곱해진다. 위 예제에서 **x**에 대한 기울기는 -8.00인데, 이는 -4.00 x 2.00이다.

> *Unintuitive effects and their consequences*. Notice that if one of the inputs to the multiply gate is very small and the other is very big, then the multiply gate will do something slightly unintuitive: it will assign a relatively huge gradient to the small input and a tiny gradient to the large input. Note that in linear classifiers where the weights are dot producted $$w^Tx_i$$ (multiplied) with the inputs, this implies that the scale of the data has an effect on the magnitude of the gradient for the weights. For example, if you multiplied all input data examples $$x_i$$ by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you’d have to lower the learning rate by that factor to compensate. This is why preprocessing matters a lot, sometimes in subtle ways! And having intuitive understanding for how the gradients flow can help you debug some of these cases.

*직관에 어긋나는 효과와 그 여파*. 곱셈 게이트의 입력 하나가 아주 작고 다른 하나가 아주 크면 곱셈 게이트가 다소 직관에 어긋나는 일을 한다는 점에 주목하자. 작은 입력에는 상대적으로 어마어마하게 큰 기울기를, 큰 입력에는 아주 작은 기울기를 준다. 가중치를 입력과 내적하는(즉 곱하는) $$w^Tx_i$$ 형태의 선형 분류기에서는, 이것이 곧 데이터의 크기가 가중치에 대한 기울기의 크기에 영향을 준다는 뜻이 된다. 예를 들어 전처리 과정에서 모든 입력 데이터 예제 $$x_i$$에 1000을 곱했다면 가중치에 대한 기울기는 1000배가 되고, 이를 상쇄하려면 학습률을 그만큼 낮춰야 한다. 그래서 전처리가 아주 중요하며, 때로는 이렇게 미묘한 방식으로 중요하다! 그리고 기울기가 어떻게 흐르는지 직관적으로 이해하고 있으면 이런 경우를 디버깅하는 데 도움이 된다.

<a id="mat"></a>

### Gradients for vectorized operations

> The above sections were concerned with single variables, but all concepts extend in a straight-forward manner to matrix and vector operations. However, one must pay closer attention to dimensions and transpose operations.

지금까지의 절들은 단일 변수를 다뤘지만, 모든 개념은 행렬과 벡터 연산으로도 그대로 확장된다. 다만 차원과 전치 연산에는 더 신경 써야 한다.

> **Matrix-Matrix multiply gradient**. Possibly the most tricky operation is the matrix-matrix multiplication (which generalizes all matrix-vector and vector-vector) multiply operations:

**행렬-행렬 곱의 기울기**. 아마 가장 까다로운 연산은 행렬-행렬 곱일 것이다(행렬-벡터 곱과 벡터-벡터 곱을 모두 일반화한다).

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

> *Tip: use dimension analysis!* Note that you do not need to remember the expressions for `dW` and `dX` because they are easy to re-derive based on dimensions. For instance, we know that the gradient on the weights `dW` must be of the same size as `W` after it is computed, and that it must depend on matrix multiplication of `X` and `dD` (as is the case when both `X,W` are single numbers and not matrices). There is always exactly one way of achieving this so that the dimensions work out. For example, `X` is of size [10 x 3] and `dD` of size [5 x 3], so if we want `dW` and `W` has shape [5 x 10], then the only way of achieving this is with `dD.dot(X.T)`, as shown above.

*요령: 차원을 따져보라!* `dW`와 `dX`의 식을 외울 필요는 없다. 차원만 보고도 쉽게 다시 유도할 수 있기 때문이다. 예를 들어 가중치에 대한 기울기 `dW`는 계산하고 나면 `W`와 크기가 같아야 하고, (`X,W`가 행렬이 아니라 숫자 하나일 때가 그렇듯) `X`와 `dD`의 행렬 곱에 의존해야 한다는 것을 안다. 차원이 맞아떨어지게 하는 방법은 언제나 정확히 하나뿐이다. 예컨대 `X`는 [10 x 3] 크기이고 `dD`는 [5 x 3] 크기인데 `W`의 모양이 [5 x 10]인 `dW`를 원한다면, 방법은 위에서 보인 `dD.dot(X.T)`뿐이다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** 차원만 맞춰도 답은 나오지만, 왜 전치가 붙는지는 원소 단위로 한 번 써보면
> 분명해진다. $$D_{ij} = \sum_k W_{ik} X_{kj}$$이므로 $$W_{ik}$$가 영향을 주는 곳은 $$D$$의
> $$i$$행 전체다. 따라서
>
> $$
> \frac{\partial L}{\partial W_{ik}} = \sum_j \frac{\partial L}{\partial D_{ij}} \frac{\partial D_{ij}}{\partial W_{ik}} = \sum_j dD_{ij} X_{kj}
> $$
>
> 이고, $$X_{kj} = (X^T)_{jk}$$이므로 이 합이 그대로 $$dW = dD \, X^T$$가 된다. 같은 방식으로
> $$dX = W^T dD$$도 나온다. 앞서 나온 '갈림길에서는 기울기가 더해진다'가 여기서는 $$j$$에
> 대한 합으로 나타난 것이다.
{: .prompt-tip }
<!-- markdownlint-restore -->

> **Work with small, explicit examples**. Some people may find it difficult at first to derive the gradient updates for some vectorized expressions. Our recommendation is to explicitly write out a minimal vectorized example, derive the gradient on paper and then generalize the pattern to its efficient, vectorized form.

**작고 구체적인 예제로 해볼 것**. 벡터화된 식의 기울기 갱신을 유도하는 일이 처음에는 어렵게 느껴질 수 있다. 권하는 방법은 최소 크기의 벡터화된 예제를 직접 써보고, 종이 위에서 기울기를 유도한 다음, 그 패턴을 효율적인 벡터화 형태로 일반화하는 것이다.

> Erik Learned-Miller has also written up a longer related document on taking matrix/vector derivatives which you might find helpful. [Find it here](http://cs231n.stanford.edu/vecDerivs.pdf).

Erik Learned-Miller도 행렬·벡터 미분에 관해 더 긴 글을 써두었으니 도움이 될 수 있다. [여기서 볼 수 있다](http://cs231n.stanford.edu/vecDerivs.pdf).

### Summary {#summary}

> - We developed intuition for what the gradients mean, how they flow backwards in the circuit, and how they communicate which part of the circuit should increase or decrease and with what force to make the final output higher.
> - We discussed the importance of **staged computation** for practical implementations of backpropagation. You always want to break up your function into modules for which you can easily derive local gradients, and then chain them with chain rule. Crucially, you almost never want to write out these expressions on paper and differentiate them symbolically in full, because you never need an explicit mathematical equation for the gradient of the input variables. Hence, decompose your expressions into stages such that you can differentiate every stage independently (the stages will be matrix vector multiplies, or max operations, or sum operations, etc.) and then backprop through the variables one step at a time.

- 기울기가 무엇을 뜻하는지, 기울기가 회로를 어떻게 거꾸로 흘러가는지, 그리고 최종 출력을 더 높이려면 회로의 어느 부분이 커지거나 작아져야 하는지를 기울기가 어느 정도의 힘으로 전달하는지에 대한 직관을 세웠다.
- 역전파를 실제로 구현할 때 **단계별 계산**이 중요하다는 것을 이야기했다. 함수는 언제나 국소 기울기를 쉽게 유도할 수 있는 모듈들로 쪼갠 다음 연쇄 법칙으로 이어붙이는 것이 좋다. 결정적으로, 이 식들을 종이에 다 적어놓고 기호적으로 완전히 미분하는 일은 거의 절대 하고 싶지 않을 것이다. 입력 변수의 기울기를 나타내는 명시적인 수식이 필요한 경우는 없기 때문이다. 그러니 각 단계를 독립적으로 미분할 수 있도록 식을 단계별로 쪼개고(각 단계는 행렬-벡터 곱이거나 max 연산이거나 합 연산 따위가 된다) 변수들을 한 번에 한 단계씩 역전파하라.

> In the next section we will start to define neural networks, and backpropagation will allow us to efficiently compute the gradient of a loss function with respect to its parameters. In other words, we’re now ready to train neural nets, and the most conceptually difficult part of this class is behind us! ConvNets will then be a small step away.

다음 절에서는 신경망을 정의하기 시작한다. 역전파 덕분에 매개변수에 대한 손실 함수의 기울기를 효율적으로 계산할 수 있게 된다. 다시 말해 이제 신경망을 학습시킬 준비가 되었고, 이 수업에서 개념적으로 가장 어려운 부분은 지나왔다! ConvNet은 거기서 작은 한 걸음이면 닿는다.

### References

> - [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)

- [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)는 기계 학습에서의 자동 미분을 개관하는 논문이다.
