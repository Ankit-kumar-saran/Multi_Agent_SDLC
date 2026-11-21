import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def main():
    out_dir = os.path.abspath("assets/screenshots")
    os.makedirs(out_dir, exist_ok=True)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1050")

    driver = webdriver.Chrome(options=options)

    try:
        print("[1/5] Navigating to platform...")
        driver.get("http://localhost:8501")
        time.sleep(3)

        # 1. Login Screen
        login_path = os.path.join(out_dir, "01_login_portal.png")
        driver.save_screenshot(login_path)
        print(f"Saved: {login_path}")

        # Fill credentials
        inputs = driver.find_elements(By.TAG_NAME, "input")
        inputs[0].send_keys("architect@bank.in")
        inputs[1].send_keys("architect123")
        for b in driver.find_elements(By.TAG_NAME, "button"):
            if "Sign In" in b.text:
                b.click()
                break
        time.sleep(4)

        # 2. Main Dashboard Landing
        dash_path = os.path.join(out_dir, "02_main_dashboard.png")
        driver.save_screenshot(dash_path)
        print(f"Saved: {dash_path}")

        # Load session with requirements
        sidebar_buttons = driver.find_elements(By.XPATH, "//section[@data-testid='stSidebar']//button")
        for b in sidebar_buttons:
            if "Low-Latency" in b.text:
                driver.execute_script("arguments[0].click();", b)
                break
        time.sleep(5)

        # 3. Requirements Analysis Top (Executive Risk, RAG Context, MoSCoW)
        driver.execute_script("document.querySelector('section.stMain').scrollTop = 0;")
        time.sleep(2)
        req_path = os.path.join(out_dir, "03_statutory_requirements_breakdown.png")
        driver.save_screenshot(req_path)
        print(f"Saved: {req_path}")

        # 4. 6-Pillars Modular Extraction & Acceptance Criteria
        driver.execute_script("document.querySelector('section.stMain').scrollTop = 750;")
        time.sleep(2)
        pillars_path = os.path.join(out_dir, "04_architectural_6_pillars.png")
        driver.save_screenshot(pillars_path)
        print(f"Saved: {pillars_path}")

        # 5. Fine-Tuned SDLC Governance Recommendation
        driver.execute_script("document.querySelector('section.stMain').scrollTop = 2200;")
        time.sleep(2)
        sdlc_path = os.path.join(out_dir, "05_fine_tuned_sdlc_recommendation.png")
        driver.save_screenshot(sdlc_path)
        print(f"Saved: {sdlc_path}")

    finally:
        driver.quit()
        print("Done!")

if __name__ == "__main__":
    main()
