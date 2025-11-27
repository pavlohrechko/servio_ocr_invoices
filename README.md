# API для Мапінгу Накладних (Invoice Mapper)

Цей сервіс використовує **Google Cloud Vision OCR** та **OpenAI LLM** для автоматичного розпізнавання накладних постачальників та співставлення (мапінгу) куплених товарів із внутрішнім списком продуктів клієнта.

Система підтримує роботу з багатьма клієнтами та запам'ятовує підтверджені відповідності для майбутніх замовлень.

---

## Швидкий старт

### 1. Налаштування

Створіть файл `.env` та додайте ключі:

```bash
OPENAI_API_KEY="sk-..."
GOOGLE_APPLICATION_CREDENTIALS="/шлях/до/ключа-google.json"
```

#Встановіть залежності:

```bash
pip install -r requirements.txt
```

# 2. Запуск сервера

```bash
flask --app app run --port 5000
```

URL API: http://127.0.0.1:5000

# Використання API (Приклади)

# 1. Завантаження списку товарів клієнта

Перед обробкою накладних необхідно завантажити список товарів (JSON масив рядків) для конкретного клієнта.

Запит:

```bash
curl -X POST [http://127.0.0.1:5001/upload-list](http://127.0.0.1:5001/upload-list) \
  -F "customer_id=cust_101" \
  -F "file=@my_items.json"
```

Де my_items.json — це файл вигляду ["Томати", "Сир", "Молоко"].

# 2. Обробка накладної

Відправляє PDF або фото накладної на розпізнавання. Повертає автоматично підтверджені товари та нові пропозиції для перевірки.

Запит:

```bash
curl -X POST [http://127.0.0.1:5001/process-invoice](http://127.0.0.1:5001/process-invoice) \
  -F "customer_id=cust_101" \
  -F "invoice=@invoice.pdf"
```

Відповідь (JSON):

```json
{
  "status": "success",
  "auto_confirmed_items": [...],
  "new_suggestions": [
    {
      "invoice_item": "Свіжі Томати 10кг",
      "suggested_menu_dish": "Томати",
      "quantity": 10.0,
      "price": 50.0
    }
  ]
}
```

# 3. Підтвердження мапінгу

Зберігає вибір користувача в пам'ять системи. Наступного разу цей товар буде розпізнано автоматично.

Запит (в один рядок):

```bash
curl -X POST [http://127.0.0.1:5001/confirm-mapping](http://127.0.0.1:5001/confirm-mapping) -H "Content-Type: application/json" -d '{"customer_id": "cust_101", "invoice_item": "Свіжі Томати 10кг", "list_item": "Томати"}'
```

Або багаторядковий варіант:

```bash
curl -X POST [http://127.0.0.1:5001/confirm-mapping](http://127.0.0.1:5001/confirm-mapping) \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_101",
    "invoice_item": "Свіжі Томати 10кг",
    "list_item": "Томати"
  }'
```

Примітка: Якщо товар не знайдено у списку, передайте null у полі list_item.
