import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox

# Global State Management
# Controls the main loop's behavior
STATE_SCANNING = 1
STATE_CHECKOUT_UI = 2
STATE_EXIT = 3
global_app_state = STATE_SCANNING

# Global Cart (uses the user's existing structure: label -> count)
cart = defaultdict(int)

# Global state for tracking ID generation
next_track_id = 1 

# --- CONFIGURATION AND CONSTANTS ---
MODEL_PATH = r"C:\Users\User\Desktop\SmartCheckout\runs2\train3\weights\best.pt"  # your new model
CAMERA_ID = 0                 # change to 1 if external webcam
FRAME_W, FRAME_H = 1280, 720         # 16:9 capture
IMG_SZ = 640                 # rectangular inference width (rect=True)
CONF_THRESH = 0.45
IOU_MATCH_THRESH = 0.3
MAX_MISSING_FRAMES = 20            # drop tracks not seen for this many frames

# checkout zone (x1,y1,x2,y2) as fraction of frame -> center-bottom rectangle by default
ZONE_FRAC = (0.10, 0.40, 0.90, 0.95) # left, top, right, bottom (fractions)

# prices
PRICES = {
  "berries gummy": 3.50,
  "chipsmore double choco": 4.20,
  "chipsmore original": 4.00,
  "dairy milk": 5.00,
  "kitkat": 3.80,
  "milo nuggetZ mocha": 6.00,
  "oreo": 2.80,
  "snickers": 3.20
}
# ----------------------------

# ---------- HELPER FUNCTIONS (Your originals) ----------
def iou_xyxy(boxA, boxB):
  # box: (x1,y1,x2,y2)
  xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
  interW = max(0, xB - xA); interH = max(0, yB - yA)
  inter = interW * interH
  areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
  areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
  union = areaA + areaB - inter
  return inter / union if union > 0 else 0

def centroid(box):
  x1,y1,x2,y2 = box
  return int((x1+x2)/2), int((y1+y2)/2)
# --------------------------------------


# --- CHECKOUT UI CLASS (TKINTER) ---

class CheckoutApp:
    def __init__(self, master, current_cart):
        self.master = master
        self.master.title("Smart Checkout")
        self.master.geometry("600x600")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # UI Cart: {item_label: tk.IntVar(quantity)}
        self.ui_cart = {label: tk.IntVar(value=qty) for label, qty in current_cart.items() if qty > 0}
        
        # Store a reference to the global cart for final update
        self.global_cart = current_cart 
        self.total_var = tk.StringVar(value="0.00")

        self.create_widgets()
        self.update_total()

    def create_widgets(self):
        # Frame for the cart list
        cart_frame = ttk.LabelFrame(self.master, text="Shopping Cart Items (Double-click item to Edit)", padding="10")
        cart_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.tree = ttk.Treeview(cart_frame, columns=("Item", "Qty", "Price", "Subtotal"), show="headings")
        
        # We need to reconfigure ALL columns now:
        
        # Set the heading for the new 'Item' column (this is now column #1)
        self.tree.heading("Item", text="Item Name", anchor=tk.W)
        self.tree.column("Item", width=250, anchor=tk.W, minwidth=250, stretch=tk.NO)
        
        # CRITICAL: We explicitly set the default column (#0) to zero width 
        # to ensure it doesn't interfere and is invisible.
        self.tree.column("#0", width=0, stretch=tk.NO) 
        
        # Configure the other columns (adjusting indices slightly)
        self.tree.heading("Qty", text="Qty", anchor=tk.CENTER)
        self.tree.column("Qty", width=60, anchor=tk.CENTER)
        
        self.tree.heading("Price", text="Price/Unit", anchor=tk.W)
        self.tree.column("Price", width=120, anchor=tk.W)
        
        self.tree.heading("Subtotal", text="Subtotal", anchor=tk.W)
        self.tree.column("Subtotal", width=120, anchor=tk.W)
        
        self.tree.pack(fill="both", expand=True)
        
        self.tree.bind("<Double-1>", self.on_item_double_click)
        
        self.load_cart_to_tree()

        # Frame for totals and buttons
        bottom_frame = ttk.Frame(self.master, padding="10")
        bottom_frame.pack(padx=10, pady=5, fill="x")

        # Total Display (Using large font for visibility)
        ttk.Label(bottom_frame, text="Grand Total:", font=('Helvetica', 12, 'normal')).pack(side="left")
        ttk.Label(bottom_frame, textvariable=self.total_var, font=('Helvetica', 18, 'bold'), foreground='black').pack(side="left", padx=10)

        # Buttons (meaningless input box removed)
        ttk.Button(bottom_frame, text="Checkout", command=self.checkout, style="Accent.TButton", width=15).pack(side="right", padx=5)
        ttk.Button(bottom_frame, text="Add More Items (Re-scan)", command=self.add_more, width=25).pack(side="right", padx=5)

    def load_cart_to_tree(self):
        """Populates the Treeview with cart data."""
        self.tree.delete(*self.tree.get_children())
        
        # Sort items by name for consistent display
        sorted_items = sorted(self.ui_cart.items())

        for label, qty_var in sorted_items:
            qty = qty_var.get()
            price = PRICES.get(label, 0.00)
            subtotal = price * qty
            
            # Use label as iid (unique identifier for Treeview item)
            # CRITICAL: We are now passing label.title() into the values tuple
            # The 'text' argument for column #0 can be an empty string, as column #0 is now hidden.
            self.tree.insert("", "end", iid=label, text="", values=(
                label.title(),  # <-- NEW: This populates the 'Item' column
                qty,            # <-- This populates the 'Qty' column
                f"RM{price:.2f}",
                f"RM{subtotal:.2f}"
            ))
        self.update_total() # Called here to ensure total is updated after loading

    def update_total(self):
        """Calculates and updates the grand total."""
        total = 0.0
        for label, qty_var in self.ui_cart.items():
            qty = qty_var.get()
            price = PRICES.get(label, 0.00)
            total += qty * price
        self.total_var.set(f"RM{total:.2f}")

    def on_item_double_click(self, event):
        """Opens a dialog to edit/remove the selected item."""
        selected_item_label = self.tree.focus()
        if not selected_item_label or selected_item_label not in self.ui_cart:
            return

        label = selected_item_label
        qty_var = self.ui_cart[label]
        
        # Pop-up window for editing
        edit_window = tk.Toplevel(self.master)
        # FIX 3: Updated the title to match the screenshot and ensure clarity
        edit_window.title(f"Edit {label.title()} Item") 
        edit_window.geometry("300x180") # FIX 4: Increased height to accommodate the buttons better
        edit_window.grab_set() # Modal behavior

        # Use a temporary variable for editing
        temp_qty_var = tk.IntVar(value=qty_var.get()) 
        
        # FIX 5: Updated label text to match the screenshot structure
        ttk.Label(edit_window, text=f"Item: {label.title()}", font=('Helvetica', 12, 'bold')).pack(pady=10)
        
        # Quantity control
        qty_frame = ttk.Frame(edit_window)
        qty_frame.pack(pady=5)
        ttk.Label(qty_frame, text="Quantity:").pack(side="left")
        
        # Spinbox for quantity (0-99)
        qty_entry = ttk.Spinbox(qty_frame, from_=0, to=99, textvariable=temp_qty_var, width=5)
        qty_entry.pack(side="left", padx=5)

        def save_changes():
            new_qty = temp_qty_var.get()
            if new_qty < 1:
                self.remove_item(label)
            else:
                qty_var.set(new_qty) # Update the actual UI cart variable
                self.load_cart_to_tree()
                self.update_total()
            edit_window.destroy()

        def remove_item_action():
            self.remove_item(label)
            edit_window.destroy()

        # Action Buttons (Fixing visibility here)
        button_frame = ttk.Frame(edit_window)
        button_frame.pack(pady=10, padx=10)
        
        ttk.Button(button_frame, text="Save", command=save_changes, width=10).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Remove Item", command=remove_item_action, style="Danger.TButton", width=15).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=edit_window.destroy, width=10).pack(side="left", padx=5)


    def remove_item(self, label):
        """Removes an item from the cart."""
        if label in self.ui_cart:
            del self.ui_cart[label]
            messagebox.showinfo("Removed", f"{label.title()} removed from cart.")
            self.load_cart_to_tree()
            self.update_total()

    def update_global_cart(self):
        """Syncs the UI cart quantities back to the global cart before continuing/exiting."""
        self.global_cart.clear()
        for label, qty_var in self.ui_cart.items():
            qty = qty_var.get()
            if qty > 0:
                self.global_cart[label] = qty

    def add_more(self):
        """Sets state to resume scanning and updates global cart."""
        global global_app_state
        self.update_global_cart()
        global_app_state = STATE_SCANNING
        self.master.destroy()

    def checkout(self):
        """Finalizes checkout, displays success message, and sets state to exit."""
        global global_app_state
        self.update_global_cart() # Final sync
        final_total = self.total_var.get()
        
        messagebox.showinfo(
            "Checkout Complete",
            f"✅ Checkout Successful!\n\nGrand Total: {final_total}\n\nThank you for shopping!"
        )
        
        global_app_state = STATE_EXIT
        self.master.destroy()

    def on_closing(self):
        """Handles window closing by setting exit state."""
        global global_app_state
        if messagebox.askyesno("Exit Checkout", "Are you sure you want to quit the program?"):
            global_app_state = STATE_EXIT
            self.master.destroy()

def run_checkout_ui(current_cart):
    """Initializes and runs the Tkinter UI."""
    root = tk.Tk()
    
    # FIX 7: Use a better-looking default theme like 'clam' or 'alt' for modern Tkinter apps
    style = ttk.Style(root)
    style.theme_use('clam') 
    
    # Define a custom style for the checkout button
    style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'), foreground='white', background='#0D9488')
    style.map("Accent.TButton", background=[('active', '#14B8A6')])
    
    # Define a custom style for the danger button
    style.configure("Danger.TButton", font=('Helvetica', 10, 'bold'), foreground='white', background='#DC2626')
    style.map("Danger.TButton", background=[('active', '#EF4444')])

    # FIX 8: Configure the Treeview font to ensure it's visible against the theme background
    style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
    style.configure("Treeview", font=('Helvetica', 10, 'normal'), rowheight=25) 

    app = CheckoutApp(root, current_cart)
    root.mainloop()


# --- SCANNING MODE (Modified from your original code) ---

def run_scanning_mode():
    """Runs the OpenCV camera loop and tracking logic."""
    global global_app_state, cart, next_track_id 
    
    # Tracking state
    tracks = {} 
    
    print("\n--- STARTING SCANNING MODE ---")

    # 1. Initialize Camera and Model
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    try:
        model = YOLO(MODEL_PATH) 
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check MODEL_PATH.")
        global_app_state = STATE_EXIT
        return

    # Compute absolute zone coordinates
    zone_x1 = int(ZONE_FRAC[0] * FRAME_W)
    zone_y1 = int(ZONE_FRAC[1] * FRAME_H)
    zone_x2 = int(ZONE_FRAC[2] * FRAME_W)
    zone_y2 = int(ZONE_FRAC[3] * FRAME_H)
    zone_rect = (zone_x1, zone_y1, zone_x2, zone_y2)

    print("Smart Checkout (tracking).")
    print("Place items through the checkout zone. Press [E] for Checkout UI. Press [Q] to quit.")
    
    frame_idx = 0
    while global_app_state == STATE_SCANNING:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting.")
            global_app_state = STATE_EXIT
            break
        frame_idx += 1

        # Run rectangular inference (preserve aspect)
        results = model(frame, imgsz=IMG_SZ, conf=CONF_THRESH, rect=True, verbose=False)
        dets = results[0].boxes # Boxes object

        # Build detection list: [(x1,y1,x2,y2,label,conf), ...]
        detections = []
        for box in dets:
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            # Handle potential CUDA tensor or NumPy array
            xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0])
            x1,y1,x2,y2 = map(int, xy)
            detections.append((x1,y1,x2,y2,label,conf))

        # match detections to existing tracks by IoU
        assigned_tracks = set()
        used_detection_idx = set()
        
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_id = None
            dbox = det[:4]
            for tid, t in tracks.items():
                if tid in assigned_tracks: 
                    continue
                i = iou_xyxy(dbox, t["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_id = tid
            if best_iou >= IOU_MATCH_THRESH and best_id is not None:
                # update track
                tracks[best_id]["bbox"] = dbox
                tracks[best_id]["label"] = det[4] # label
                tracks[best_id]["missing_frames"] = 0
                tracks[best_id]["last_centroid"] = centroid(dbox)
                assigned_tracks.add(best_id)
                used_detection_idx.add(det_idx)

        # add unmatched detections as new tracks
        for det_idx, det in enumerate(detections):
            if det_idx in used_detection_idx:
                continue
            dbox = det[:4]
            label = det[4]
            tracks[next_track_id] = {
                "bbox": dbox,
                "label": label,
                "entered": False,
                "counted": False,
                "missing_frames": 0,
                "last_centroid": centroid(dbox)
            }
            next_track_id += 1

        # increment missing_frames for unassigned tracks
        to_delete = []
        for tid, t in list(tracks.items()):
            if tid not in assigned_tracks:
                t["missing_frames"] += 1
                if t["missing_frames"] > MAX_MISSING_FRAMES:
                    # remove old track
                    to_delete.append(tid)

        # remove stale tracks
        for tid in to_delete:
            del tracks[tid]

        # Process tracking logic (enter/exit zone and counting)
        for tid, t in tracks.items():
            x1,y1,x2,y2 = t["bbox"]
            cx, cy = t["last_centroid"]
            inside = (zone_x1 <= cx <= zone_x2) and (zone_y1 <= cy <= zone_y2)

            # detect entering
            if inside and not t["entered"]:
                t["entered"] = True
            # detect exiting after previously entered (count on exit)
            if (not inside) and t["entered"] and (not t["counted"]):
                # count it
                cart[t["label"]] += 1
                t["counted"] = True

        # Draw results on frame
        # draw zone
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
        cv2.putText(frame, "CHECKOUT ZONE", (zone_x1+5, zone_y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # draw tracks
        for tid, t in tracks.items():
            x1,y1,x2,y2 = t["bbox"]
            label = t["label"]
            cx, cy = t["last_centroid"]
            color = (0,255,0) if t.get("counted") else (0,200,200)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID{tid} {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx,cy), 3, color, -1)

        # draw cart summary on left
        y = 30
        cv2.putText(frame, "Cart:", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        y += 28
        total = 0.0
        # Only show items with quantity > 0
        current_cart_items = {k: v for k, v in cart.items() if v > 0} 
        for k, v in current_cart_items.items():
            price = PRICES.get(k, 0.0)
            subtotal = price * v
            cv2.putText(frame, f"{k} x{v} = RM{subtotal:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            y += 22
            total += subtotal
        cv2.putText(frame, f"Total: RM{total:.2f}", (10, FRAME_H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        cv2.putText(frame, "Press 'E' for Checkout UI, 'Q' to Quit.", (FRAME_W - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Smart Checkout (tracking - 16:9)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('e'):
            global_app_state = STATE_CHECKOUT_UI
            print("Entering Checkout UI...")
            break
        elif key == ord('q'):
            global_app_state = STATE_EXIT
            print("Exiting application...")
            break

    # Cleanup resources after the scanning loop ends
    cap.release()
    cv2.destroyAllWindows()


# --- MAIN APPLICATION STATE MANAGER ---

def main_app():
    """Manages the application state (Scanning or Checkout UI)."""
    global global_app_state, cart

    while global_app_state != STATE_EXIT:
        
        if global_app_state == STATE_SCANNING:
            run_scanning_mode() # This function blocks until 'E' or 'Q' is pressed
            
        elif global_app_state == STATE_CHECKOUT_UI:
            # We filter the cart to only include items with quantity > 0
            current_valid_cart = {k: v for k, v in cart.items() if v > 0}
            if not current_valid_cart:
                 # If cart is empty, skip UI and go back to scanning
                 messagebox.showinfo("Cart Empty", "The cart is currently empty. Returning to scanning mode.")
                 global_app_state = STATE_SCANNING
                 continue
                 
            run_checkout_ui(cart) # This function blocks until UI is closed/button is pressed
            
        # The main while loop continues based on the updated global_app_state

    print("Application closed.")


if __name__ == "__main__":
    main_app()