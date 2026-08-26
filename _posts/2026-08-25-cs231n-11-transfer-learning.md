---
title: "11. Transfer Learning and Fine-tuning Convolutional Neural Networks"
description: "사전 학습된 ConvNet을 특징 추출기로 쓰거나 fine-tuning하는 전략과 실무 지침."
date: 2026-08-25 09:50:00 +0900
categories: [Computer Vision, cs231n]
tags: [study, computer vision, cs231n, deep learning]
math: true
---

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **원문**: [Transfer Learning and Fine-tuning Convolutional Neural Networks](https://cs231n.github.io/transfer-learning/)
> — CS231n: Convolutional Neural Networks for Visual Recognition (Stanford University) · © 2015 Andrej Karpathy, MIT License
>
> 원문을 문단 단위로 인용하고 그 아래에 한국어 번역을 붙였다. 인용 블록이 원문, 그 아래 문단이 번역이며, `역주` 박스와 `보충` 섹션은 원문에 없는 추가 내용이다.
{: .prompt-info }
<!-- markdownlint-restore -->

> (These notes are currently in draft form and under development)

(이 노트는 현재 초안 상태이며 계속 작성되고 있다.)

> Table of Contents:

목차는 다음과 같다.

> - [Transfer Learning](#tf)
> - [Additional References](#add)

- [전이 학습](#tf)
- [추가 참고 자료](#add)

<span id="tf"></span>

## Transfer Learning

> In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. The three major Transfer Learning scenarios look as follows:

실제로는 ConvNet 전체를(무작위 초기화로) 처음부터 학습시키는 사람이 매우 드문데, 충분한 크기의 데이터셋을 갖추는 경우가 상대적으로 드물기 때문이다. 그 대신 아주 큰 데이터셋(예컨대 120만 장의 이미지와 1000개의 범주로 이루어진 ImageNet)으로 ConvNet을 미리 학습시켜 두고, 그 ConvNet을 관심 있는 과제를 위한 초기화 값이나 고정된 특징 추출기(feature extractor)로 쓰는 것이 일반적이다. 전이 학습(Transfer Learning)의 주요 시나리오 세 가지는 다음과 같다.

> - **ConvNet as fixed feature extractor**. Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features **CNN codes**. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.
> - **Fine-tuning the ConvNet**. The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.
> - **Pretrained models**. Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, the Caffe library has a [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) where people share their network weights.

- **고정된 특징 추출기로 쓰는 ConvNet**. ImageNet으로 미리 학습된 ConvNet을 가져와 마지막 완전 연결 층을(이 층의 출력이 ImageNet 같은 다른 과제를 위한 1000개의 클래스 점수다) 제거한 뒤, ConvNet의 나머지 부분을 새 데이터셋을 위한 고정된 특징 추출기로 취급한다. AlexNet이라면 분류기 바로 앞 은닉층의 활성값으로 이루어진 4096차원 벡터를 이미지마다 계산하게 된다. 이 특징들을 **CNN 코드(CNN codes)**라고 부른다. ImageNet에서 ConvNet을 학습시킬 때도 보통 그렇듯 이 코드들에 문턱 처리를 했다면, 여기서도 이 코드들이 ReLU를 거친(곧 0에서 문턱 처리된) 값이어야 한다는 점이 성능에 중요하다. 모든 이미지에 대해 4096차원 코드를 뽑아낸 뒤에는, 새 데이터셋을 위한 선형 분류기(예컨대 Linear SVM이나 Softmax 분류기)를 학습시키면 된다.
- **ConvNet fine-tuning하기**. 두 번째 전략은 새 데이터셋에 맞춰 ConvNet 위의 분류기를 교체하고 다시 학습시키는 데서 그치지 않고, 역전파를 계속 이어가며 미리 학습된 신경망의 가중치까지 fine-tuning하는 것이다. ConvNet의 모든 층을 fine-tuning할 수도 있고, (과적합 우려 때문에) 앞쪽의 몇몇 층은 고정해 둔 채 신경망의 상위 일부만 fine-tuning할 수도 있다. 이런 방식을 택하는 근거는, ConvNet의 앞쪽 특징일수록 여러 과제에 두루 쓸모 있는 좀 더 일반적인(generic) 특징(예컨대 에지 검출기나 색 얼룩 검출기)을 담고 있는 반면, ConvNet의 뒤쪽 층으로 갈수록 원래 데이터셋에 담긴 클래스의 세부 사항에 점점 더 특화된다는 관찰이다. 예컨대 개 품종이 많이 들어 있는 ImageNet의 경우, ConvNet의 표현력 가운데 상당 부분이 개 품종을 구별하는 데 특화된 특징에 쓰이고 있을 수 있다.
- **미리 학습된 모델**. 요즘 ConvNet은 여러 GPU를 동원해도 ImageNet을 학습시키는 데 2~3주가 걸리기 때문에, 사람들이 최종 ConvNet 체크포인트를 공개해서 다른 사람이 그 신경망을 fine-tuning에 쓸 수 있게 하는 경우를 흔히 볼 수 있다. 예컨대 Caffe 라이브러리에는 사람들이 신경망 가중치를 공유하는 [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)가 있다.

> **When and how to fine-tune?** How do you decide what type of transfer learning you should perform on a new dataset? This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

**언제, 어떻게 fine-tuning할 것인가?** 새 데이터셋에 어떤 종류의 전이 학습을 적용할지는 어떻게 정할까? 여러 요인에 달려 있지만, 그중 가장 중요한 두 가지는 새 데이터셋의 크기(작은지 큰지)와 원래 데이터셋과의 유사성(이미지 내용이나 클래스가 ImageNet과 비슷한지, 아니면 현미경 이미지처럼 아주 다른지)이다. ConvNet의 특징이 앞쪽 층일수록 더 일반적이고 뒤쪽 층일수록 원래 데이터셋에 더 특화된다는 점을 염두에 두고, 주요 시나리오 4가지를 헤쳐나가는 데 흔히 쓰는 어림법을 소개한다.

> 1. *New dataset is small and similar to original dataset*. Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
> 2. *New dataset is large and similar to the original dataset*. Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.
> 3. *New dataset is small but very different from the original dataset*. Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.
> 4. *New dataset is large and very different from the original dataset*. Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

1. *새 데이터셋이 작고 원래 데이터셋과 비슷할 때*. 데이터가 작으므로 과적합 우려 때문에 ConvNet을 fine-tuning하는 것은 좋은 생각이 아니다. 데이터가 원래 데이터와 비슷하므로 ConvNet의 상위 특징들도 이 데이터셋에 여전히 들어맞으리라 기대할 수 있다. 따라서 CNN 코드에 대해 선형 분류기를 학습시키는 것이 가장 좋은 방법일 수 있다.
2. *새 데이터셋이 크고 원래 데이터셋과 비슷할 때*. 데이터가 더 많으므로 신경망 전체를 fine-tuning하더라도 과적합되지 않으리라는 확신을 더 크게 가질 수 있다.
3. *새 데이터셋이 작고 원래 데이터셋과 매우 다를 때*. 데이터가 작으므로 선형 분류기만 학습시키는 편이 가장 나을 가능성이 높다. 데이터셋이 매우 다르므로 데이터셋에 좀 더 특화된 특징을 담고 있는 신경망 꼭대기에서 분류기를 학습시키는 것은 최선이 아닐 수 있다. 그 대신 신경망의 좀 더 앞쪽 어딘가의 활성값으로 SVM 분류기를 학습시키는 편이 더 잘 통할 수 있다.
4. *새 데이터셋이 크고 원래 데이터셋과 매우 다를 때*. 데이터셋이 매우 크므로 ConvNet을 처음부터 학습시킬 여유가 있으리라 기대할 수 있다. 그렇지만 실제로는 미리 학습된 모델의 가중치로 초기화하는 편이 여전히 유리한 경우가 아주 많다. 이 경우라면 신경망 전체를 fine-tuning할 만큼 충분한 데이터와 확신을 갖게 될 것이다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** 이 네 시나리오를 가르는 두 축은 "데이터가 얼마나 많은가"와 "원래 데이터셋과 얼마나 비슷한가"다. 데이터가 적으면 매개변수를 많이 건드릴수록 과적합 위험이 커지므로 되도록 적게 건드리고(CNN 코드 위에 선형 분류기만), 데이터가 많으면 그 위험이 줄어들어 더 깊이까지 건드릴 여유가 생긴다. 원래 데이터셋과 비슷하면 뒤쪽 층 특징을 그대로 재사용해도 되지만, 많이 다르면 뒤쪽 층 특징이 원래 데이터셋의 클래스에 특화되어 있어 오히려 방해가 될 수 있으므로 더 일반적인 특징이 남아 있는 앞쪽 층에서 시작하는 편이 낫다.
{: .prompt-tip }
<!-- markdownlint-restore -->

> **Practical advice**. There are a few additional things to keep in mind when performing Transfer Learning:

**실무 조언**. 전이 학습을 수행할 때 유념해 둘 만한 몇 가지가 더 있다.

> - *Constraints from pretrained models*. Note that if you wish to use a pretrained network, you may be slightly constrained in terms of the architecture you can use for your new dataset. For example, you can’t arbitrarily take out Conv layers from the pretrained network. However, some changes are straight-forward: Due to parameter sharing, you can easily run a pretrained network on images of different spatial size. This is clearly evident in the case of Conv/Pool layers because their forward function is independent of the input volume spatial size (as long as the strides “fit”). In case of FC layers, this still holds true because FC layers can be converted to a Convolutional Layer: For example, in an AlexNet, the final pooling volume before the first FC layer is of size [6x6x512]. Therefore, the FC layer looking at this volume is equivalent to having a Convolutional Layer that has receptive field size 6x6, and is applied with padding of 0.
> - *Learning rates*. It’s common to use a smaller learning rate for ConvNet weights that are being fine-tuned, in comparison to the (randomly-initialized) weights for the new linear classifier that computes the class scores of your new dataset. This is because we expect that the ConvNet weights are relatively good, so we don’t wish to distort them too quickly and too much (especially while the new Linear Classifier above them is being trained from random initialization).

- *미리 학습된 모델이 거는 제약*. 미리 학습된 신경망을 쓰고 싶다면 새 데이터셋에 쓸 수 있는 구조가 다소 제약될 수 있다는 점에 주의하자. 예컨대 미리 학습된 신경망에서 Conv 층을 마음대로 빼낼 수는 없다. 그렇지만 손쉬운 변경도 있다. 매개변수 공유 덕분에 서로 다른 공간 크기의 이미지에서도 미리 학습된 신경망을 쉽게 돌릴 수 있다. Conv/Pool 층의 경우 순전파 함수가 입력 부피의 공간 크기와 무관하므로(stride가 "딱 들어맞는" 한) 이 점이 뚜렷하게 드러난다. FC 층의 경우도 FC 층을 합성곱 층으로 바꿀 수 있으므로 마찬가지다. 예컨대 AlexNet에서 첫 FC 층 앞의 마지막 pooling 부피는 크기가 [6x6x512]다. 따라서 이 부피를 들여다보는 FC 층은 수용 영역 크기가 6x6이고 padding 0으로 적용되는 합성곱 층을 두는 것과 동등하다.
- *학습률*. 새 데이터셋의 클래스 점수를 계산하는(무작위로 초기화된) 새 선형 분류기의 가중치에 비해, fine-tuning되고 있는 ConvNet의 가중치에는 더 작은 학습률을 쓰는 것이 일반적이다. ConvNet의 가중치는 이미 상당히 좋은 상태이리라 기대하므로, (특히 그 위의 새 선형 분류기가 무작위 초기화에서부터 학습되는 동안) 그 가중치를 너무 빠르고 크게 흐트러뜨리고 싶지 않기 때문이다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> **역주.** 학습률을 다르게 주는 이유를 한 줄로 요약하면, 이미 잘 잡혀 있는 가중치는 살살 다듬고 아직 무작위인 가중치는 크게 움직여야 한다는 것이다. 새 선형 분류기는 무작위 초기화에서 출발하므로 큰 학습률로 빠르게 제 위치를 찾아가야 하지만, 그 아래 ConvNet의 가중치는 이미 유용한 특징을 담고 있으므로 같은 학습률로 갱신하면 그 특징이 금세 뭉개질 수 있다. 그래서 실무에서는 흔히 새 분류기 쪽 학습률을 기준으로 잡고, ConvNet 쪽 학습률은 그보다 더 작은 값으로 따로 설정한다.
{: .prompt-tip }
<!-- markdownlint-restore -->

<span id="add"></span>

## Additional References

> - [CNN Features off-the-shelf: an Astounding Baseline for Recognition](http://arxiv.org/abs/1403.6382) trains SVMs on features from ImageNet-pretrained ConvNet and reports several state of the art results.
> - [DeCAF](http://arxiv.org/abs/1310.1531) reported similar findings in 2013. The framework in this paper (DeCAF) was a Python-based precursor to the C++ Caffe library.
> - [How transferable are features in deep neural networks?](http://arxiv.org/abs/1411.1792) studies the transfer learning performance in detail, including some unintuitive findings about layer co-adaptations.

- [CNN Features off-the-shelf: an Astounding Baseline for Recognition](http://arxiv.org/abs/1403.6382)는 ImageNet으로 미리 학습된 ConvNet의 특징으로 SVM을 학습시켜 여러 최고 성능 결과를 보고한다.
- [DeCAF](http://arxiv.org/abs/1310.1531)는 2013년에 비슷한 결과를 보고했다. 이 논문(DeCAF)의 프레임워크는 C++로 된 Caffe 라이브러리보다 앞서 나온, Python 기반의 선행 프레임워크였다.
- [How transferable are features in deep neural networks?](http://arxiv.org/abs/1411.1792)는 층 간 공적응(co-adaptation)에 관한 다소 직관에 어긋나는 발견을 포함해 전이 학습의 성능을 자세히 연구한다.
