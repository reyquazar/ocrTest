def predictModel():
    from ultralytics import YOLO
    import albumentations


    model = YOLO("runs/best6.pt")

    # results = model.predict(source="coco8/images/train/100.jpg", save=True, conf=0.25)
    # results = model.predict(source="testimages/img.png", save=True, conf=0.25)
    results = model.predict(source="captcha/captcha_83vd7.png", save=True, conf=0.25)

def learnTorch():
    import torch
    import torch.nn as nn

    # Простая сеть: 1 вход → 1 выход
    модель = nn.Linear(1, 1)

    x = torch.tensor([[3.0]])
    правильный_ответ = torch.tensor([[15.0]])  # ← твоё число 15

    loss_fn = nn.MSELoss()
    оптимизатор = torch.optim.SGD(модель.parameters(), lr=0.01)  # ← было 0.1, стало 0.01

    print("До обучения:", модель(x).item())

    for шаг in range(50):
        предсказание = модель(x)
        ошибка = loss_fn(предсказание, правильный_ответ)

        оптимизатор.zero_grad()
        ошибка.backward()
        оптимизатор.step()
        print(f"Шаг {шаг}: предсказание = {предсказание.item():.2f}, ошибка = {ошибка.item():.2f}")

    print("После обучения:", модель(x).item())


predictModel()
# learnTorch()