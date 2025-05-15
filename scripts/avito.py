import time
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium_stealth import stealth

options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome()

stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )

url = "https://www.avito.ru/sankt-peterburg/avtomobili/bmw-ASgBAgICAUTgtg3klyg?cd=1&f=ASgBAgICAkTgtg3klyiwsxT~oY8D&p=43&radius=0&searchRadius=0"
driver.get(url)

page_limit = 10
current_page = 1
number_of_car = 1
car_data = []

while current_page <= page_limit:
    car_links = driver.find_elements(By.XPATH, "//a[@data-marker='item-title']")
    print(f"Парсинг страницы каталога {current_page}")
    for link in car_links:
        actions = ActionChains(driver)
        actions.key_down(Keys.CONTROL).click(link).key_up(Keys.CONTROL).perform()
        driver.switch_to.window(driver.window_handles[1])
        time.sleep(3)
        open_data = False

        print(f"Парсинг машины {number_of_car}")
        try:
            car_details = {}

            number_of_car += 1

            price_value = (driver.find_element(By.XPATH, '//span[@data-marker="item-view/item-price"]')).get_attribute(
                'content')
            car_name = (driver.find_element(By.XPATH, "//h1[@data-marker='item-view/title-info']").text.split(',')[0])
            content = driver.find_elements(By.XPATH, '//li[@class="params-paramsList__item-_2Y2O"]')
            year = (content[0].text).split()[-1]
            mileage = "".join((content[2].text).split()[1:-1])

            all_specs_button = driver.find_element(By.XPATH, "//a[@data-marker='item-specification-button']")
            actions.move_to_element(all_specs_button).click().perform()
            time.sleep(1)

            driver.switch_to.window(driver.window_handles[2])
            open_data = True
            time.sleep(3)

            elements = driver.find_elements(By.XPATH,
                                            '//dd[@class="styles-module-column-_7N7q styles-module-column_span_6-XlUy_"]/span')

            values = [elements[element].text.strip() for element in range(7)]

            car_details['Name'] = car_name
            car_details['Year'] = year
            car_details['Mileage'] = mileage
            car_details['Engine_capacity'] = values[0]
            car_details['Engine_type'] = values[1]
            car_details['Power'] = values[2]
            car_details['Transmission'] = values[3]
            car_details['Drive'] = values[4]
            car_details['Fuel_consumption_mixed'] = values[5]
            car_details['Acceleration_to_100'] = values[6]
            car_details['Cost'] = price_value
            car_data.append(car_details)

        except Exception as e:
            print(f"Ошибка при открытии характеристик машины номер {number_of_car}")

        all_tabs = driver.window_handles

        for tab in all_tabs[1:]:
            driver.switch_to.window(tab)
            time.sleep(2)
            driver.close()

        driver.switch_to.window(all_tabs[0])
    try:
        next_button = driver.find_element(By.XPATH, "//a[@data-marker='pagination-button/nextPage']")
        actions.move_to_element(next_button).click().perform()
        current_page += 1
        time.sleep(3)

        driver.get(driver.current_url)
        time.sleep(3)

    except:
        print("Достигли последней страницы")
        break

df = pd.DataFrame(car_data)
df.to_csv("out.csv", index=False)
print("Парсинг завершён!")
driver.quit()
df.head(3)
