
import tkinter as tk
import webview

# Create main window
root = tk.Tk()
root.title("Mini YouTube")
root.geometry("560x315")  # Small window size

# Open YouTube in a webview
webview.create_window('Mini YouTube', 'https://www.youtube.com')
webview.start()

root.mainloop()
