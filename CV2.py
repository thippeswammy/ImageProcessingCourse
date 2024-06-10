import cv2
import tkinter as tk

# Create a temporary Tkinter window (hidden)
root = tk.Tk()
root.withdraw()  # Hide the win
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()


def imshowC(winname: str, mat: cv2.typing.MatLike) -> None:
    cv2.imshow(winname, mat)
    cv2.moveWindow(winname, (screen_width - mat.shape[1]) // 2, (screen_height - mat.shape[0]) // 2)
    cv2.waitKey(0)


def imshow(winname: str, mat: cv2.typing.MatLike) -> None:
    cv2.imshow(winname, mat)
    cv2.moveWindow(winname, (screen_width - mat.shape[1]) // 2, (screen_height - mat.shape[0]) // 2)
    cv2.waitKey(0)
