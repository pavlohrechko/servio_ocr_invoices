# API для Мапінгу Накладних (Invoice Mapper)

Цей сервіс використовує **Google Gemini** для автоматичного розпізнавання накладних постачальників та співставлення (мапінгу) куплених товарів із внутрішнім списком продуктів клієнта.

Система підтримує роботу з багатьма клієнтами та запам'ятовує підтверджені відповідності для майбутніх замовлень.

---

## Швидкий старт

### 1. Налаштування

Створіть файл `.env` та додайте ключі:

```bash
GEMINI_API_KEY="your-gemini-api-key"
JWT_SECRET_KEY="your-secret-key"
```

Встановіть залежності:

```bash
pip install -r requirements.txt
```

### 2. Запуск сервера

```bash
# Режим розробки
python app.py

# Продакшн (рекомендовано)
gunicorn -w 3 -k gevent -b 0.0.0.0:5000 --timeout 300 app:app
```

URL API: `http://127.0.0.1:5000`

---

## Автентифікація

Всі endpoint-и захищені JWT токенами. Перед використанням API необхідно зареєструватись та отримати токен.

### Реєстрація користувача

```bash
curl -X POST http://127.0.0.1:5000/oauth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "yourpassword"}'
```

Відповідь:

```json
{
  "status": "success",
  "message": "User 'user@example.com' created."
}
```

### Отримання токену

```bash
curl -X POST http://127.0.0.1:5000/oauth/token \
  -H "Content-Type: application/json" \
  -d '{"grant_type": "password", "username": "user@example.com", "password": "yourpassword"}'
```

Відповідь:

```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

| Токен           | Термін дії | Призначення                                        |
| --------------- | ---------- | -------------------------------------------------- |
| `access_token`  | 1 година   | Додається до кожного запиту                        |
| `refresh_token` | 30 днів    | Отримання нового access_token без повторного входу |

### Оновлення токену (без повторного входу)

```bash
curl -X POST http://127.0.0.1:5000/oauth/token \
  -H "Content-Type: application/json" \
  -d '{"grant_type": "refresh_token", "refresh_token": "eyJ..."}'
```

### Використання токену в запитах

Додайте заголовок `Authorization` до кожного запиту:

```bash
curl -X POST http://127.0.0.1:5000/process-invoice \
  -H "Authorization: Bearer eyJ..." \
  -F "customer_id=cust_101" \
  -F "invoice=@invoice.pdf"
```

---

## Підтримувані формати файлів

Endpoint `/process-invoice` підтримує наступні формати:

| Формат     | Розширення              | Примітка                         |
| ---------- | ----------------------- | -------------------------------- |
| Зображення | `.jpg`, `.jpeg`, `.png` | Фото накладної                   |
| HEIC       | `.heic`                 | Фото з iPhone                    |
| PDF        | `.pdf`                  | Сканований або цифровий документ |
| Excel      | `.xlsx`, `.xls`         | Таблиці Excel                    |
| CSV        | `.csv`                  | Текстові таблиці                 |
| Word       | `.docx`, `.doc`         | Документи Microsoft Word         |

Максимальний розмір файлу: **32 МБ**

---

## Використання API

### 1. Завантаження списку товарів клієнта

Перед обробкою накладних необхідно завантажити список товарів (JSON масив рядків) для конкретного клієнта.

```bash
curl -X POST http://127.0.0.1:5000/upload-list \
  -H "Authorization: Bearer eyJ..." \
  -F "customer_id=cust_101" \
  -F "file=@my_items.json"
```

Де `my_items.json` — файл вигляду:

```json
["Томати", "Сир", "Молоко", "Хліб"]
```

Відповідь:

```json
{
  "status": "success",
  "message": "List for 'cust_101' saved successfully.",
  "item_count": 4
}
```

### 2. Обробка накладної

Відправляє файл накладної на розпізнавання. Повертає автоматично підтверджені товари та нові пропозиції для перевірки.

```bash
curl -X POST http://127.0.0.1:5000/process-invoice \
  -H "Authorization: Bearer eyJ..." \
  -F "customer_id=cust_101" \
  -F "invoice=@invoice.pdf"
```

Відповідь (JSON):

```json
{
  "status": "success",
  "processing_time_sec": 3.17,
  "auto_confirmed_items": [...],
  "new_suggestions": [
    {
      "invoice_item": "Свіжі Томати 10кг",
      "suggested_item": "Томати",
      "quantity": 10.0,
      "unit": "кг",
      "price": 50.0,
      "price_nds": 60.0,
      "amount": 500.0,
      "amount_nds": 600.0,
      "nds": "20%",
      "product_code": "001",
      "notes": null
    }
  ]
}
```

### 3. Підтвердження мапінгу

Зберігає вибір користувача. Наступного разу цей товар буде розпізнано автоматично.

```bash
curl -X POST http://127.0.0.1:5000/confirm-mapping \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_101",
    "invoice_item": "Свіжі Томати 10кг",
    "list_item": "Томати"
  }'
```

> **Примітка:** Якщо товар не знайдено у списку, передайте `null` у полі `list_item`.

---

## Поля відповіді

| Поле             | Опис                                |
| ---------------- | ----------------------------------- |
| `invoice_item`   | Назва товару з накладної            |
| `suggested_item` | Відповідний товар зі списку клієнта |
| `quantity`       | Кількість                           |
| `unit`           | Одиниця виміру                      |
| `price`          | Ціна без ПДВ                        |
| `price_nds`      | Ціна з ПДВ                          |
| `amount`         | Сума без ПДВ                        |
| `amount_nds`     | Сума з ПДВ                          |
| `nds`            | Ставка ПДВ (наприклад `"20%"`)      |
| `product_code`   | Код товару постачальника            |
| `notes`          | Додаткова інформація                |
