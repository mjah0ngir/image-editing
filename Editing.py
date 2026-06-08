
import cv2
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3"


def ask_llm(user_text):
    prompt = f"""

Ты классификатор команд для обработки изображений.
Отвечай ТОЛЬКО одним словом.
Доступные команды:

rotate
rotate_left
gray
blur
flip
brightness
contrast
invert
edges
sketch

Команда пользователя:
{user_text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    command = response.json()["response"].strip().lower()

    if command not in {
        "rotate",
        "rotate_left",
        "gray",
        "blur",
        "flip",
        "brightness",
        "contrast",
        "invert",
        "edges",
        "sketch"
    }:
        command = command.split()[0]

    return command

def process_image(img, command):

    if command == "rotate":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif command == "rotate_left":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif command == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif command == "blur":
        return cv2.GaussianBlur(img, (15, 15), 0)

    elif command == "flip":
        return cv2.flip(img, 1)

    elif command == "brightness":
        return cv2.convertScaleAbs(img, alpha=1, beta=50)

    elif command == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.7, beta=0)

    elif command == "invert":
        return cv2.bitwise_not(img)

    elif command == "edges":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    elif command == "sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        return cv2.divide(gray, 255 - blur, scale=256)

    return img

def main():

    image_path = input("Путь к изображению: ").strip()

    img = cv2.imread(image_path)

    if img is None:
        print("Ошибка: изображение не найдено.")
        return

    user_command = input("Введите команду: ").strip()

    command = ask_llm(user_command)

    print(f"Нейросеть определила команду: {command}")

    result = process_image(img, command)

    cv2.imwrite("result.jpg", result)

    cv2.imshow("Результат", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Результат сохранён в result.jpg")

if __name__ == "__main__":
    main()