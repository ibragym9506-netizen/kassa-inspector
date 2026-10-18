"""
Справочник нарушений. Составлен по разбору датасета (docs/datathon_analytics.xlsx).

ML-часть возвращает только КЛЮЧ нарушения (kind), таймкод и кадры.
Название, описание и норматив сервис берёт отсюда, поэтому ссылки всегда
проверенные, а не придуманные моделью.

Нормативы сверены с текстами на adilet.zan.kz / zakon.uchet.kz 17.10.2026.
Где точного пункта нет, указан только документ (ТЗ это разрешает).

НЕ СЧИТАТЬ НАРУШЕНИЕМ (по разметке организаторов — будут ложные срабатывания):
  - клиент не предъявил документ (Х-19, Х-35, Х-37);
  - смартфон у кассира (Х-10);
  - дверь закрыта, но не заперта на ключ (Х-12);
  - горящий экран ноутбука, если рабочее приложение закрыто (Х-23, Х-29);
  - свет оставлен включённым (Х-13, Х-23);
  - кассир одна, без контролёра (Х-8, Х-14);
  - деньги на столе, пока кассир с ними работает (Х-8);
  - проверенные купюры сложены в одну стопку (Х-45).
"""

R231 = "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей (постановление Правления НБРК от 29.11.2019 № 231)"
R14 = "Правила организации охраны и устройства помещений банков второго уровня (постановление Правления НБРК от 24.02.2020 № 14)"
RIB = "Требования к обеспечению информационной безопасности банков (постановление Правления АРРФР от 03.04.2026 № 53)"

RULES = {
    "valuables_left": {
        "type": "Ценности оставлены без присмотра",
        "description": "Кассир покидает кассу, оставив на столе ценности (наличные деньги, опломбированный пакет, документы или штампы), не убрав их в сейф.",
        "hint": "the cashier leaves the room while cash, a sealed money bag, documents or stamps stay on the desk (not put into the safe)",
        "rule": {"document": R231, "clause": "п. 20; для штампов — п. 80"},
    },
    "prev_client_money": {
        "type": "Деньги предыдущего клиента не убраны",
        "description": "Кассир начинает работу со следующим клиентом, а деньги предыдущего остаются на столе или складываются в одну стопку с новыми.",
        "hint": "a new customer is served while the previous customer's banknotes still lie on the desk or are put into the same stack",
        "rule": {"document": R231, "clause": "п. 20"},
    },
    "door_open": {
        "type": "Дверь кассы открыта",
        "description": "Кассир находится в кассе, дверь кассового помещения не закрыта; в помещение может войти или заглянуть посторонний.",
        "hint": "the cashier is inside and the door of the cashier room is left open (a closed but unlocked door is fine)",
        "rule": {"document": R14, "clause": "п. 8"},
    },
    "app_left_open": {
        "type": "Рабочее приложение оставлено открытым",
        "description": "Кассир покидает рабочее место, оставив на компьютере открытое рабочее приложение без завершения сеанса.",
        "hint": "the cashier leaves while a work program window stays open on the laptop screen (a lit screen with no program open is fine)",
        "rule": {"document": RIB, "clause": ""},
    },
    "safe_open": {
        "type": "Сейф оставлен открытым",
        "description": "Во время работы с деньгами дверца сейфа остаётся открытой.",
        "hint": "the safe door stays open while the cashier works with money",
        "rule": {"document": R231, "clause": "п. 20"},
    },
    "key_in_safe": {
        "type": "Ключ оставлен в замке сейфа",
        "description": "Кассир закрывает сейф, но оставляет ключ в замке и покидает кассу.",
        "hint": "the cashier closes the safe but leaves the key in its lock and walks away",
        "rule": {"document": R231, "clause": "п. 20, п. 76"},
    },
    "count_without_client": {
        "type": "Пересчёт денег без клиента",
        "description": "Клиент уходит из кассы до окончания пересчёта его денег; пересчёт продолжается без клиента.",
        "hint": "the customer leaves before the money counting is finished; counting continues without the customer",
        "rule": {"document": R231, "clause": "п. 20"},
    },
    "parallel_service": {
        "type": "Параллельное обслуживание двух посетителей",
        "description": "Кассир начинает обслуживать второго посетителя (принимает документы), пока операция с первым не завершена.",
        "hint": "the cashier takes documents or money from a second visitor while the first customer's operation is not finished",
        "rule": {"document": R231, "clause": ""},
    },
    "bag_without_docs": {
        "type": "Пакет принят без сопроводительных документов",
        "description": "Кассир принимает опломбированный пакет, при этом сопроводительные документы не передаются и не проверяются.",
        "hint": "a sealed money bag is handed to the cashier without any documents being passed and checked first",
        "rule": {"document": R231, "clause": "п. 24"},
    },
    "no_authenticity_check": {
        "type": "Не проведена проверка подлинности купюр",
        "description": "После пересчёта на машине купюры сразу складываются в стопку, без проверки под ультрафиолетом и ручного пересчёта.",
        "hint": "after the counting machine the banknotes go straight into a stack, with no UV-lamp check and no manual recount",
        "rule": {"document": R231, "clause": "пп. 57–58"},
    },
    "bag_to_safe_uncounted": {
        "type": "Деньги из вскрытой упаковки убраны без пересчёта",
        "description": "Кассир вскрывает опломбированную упаковку и убирает наличные в сейф без пересчёта на машине.",
        "hint": "the cashier opens a sealed bag and puts the cash into the safe without counting it in the machine",
        "rule": {"document": R231, "clause": "п. 25, пп. 57–58"},
    },
    "personal_items": {
        "type": "Личные вещи на рабочем месте",
        "description": "На рабочем столе кассира, где ведётся работа с наличными, находятся личные вещи (сумка, верхняя одежда).",
        "hint": "a handbag, backpack or jacket lies on the cashier's work desk",
        "rule": {"document": R231, "clause": "п. 12"},
    },
    "food_near_cash": {
        "type": "Еда и напитки рядом с наличными",
        "description": "Кассир ест или пьёт на рабочем месте, при этом наличные деньги лежат на столе и не убраны.",
        "hint": "the cashier eats or drinks at the desk while counted cash lies on the desk",
        "rule": {"document": R231, "clause": "п. 20"},
    },
    "phone_during_cash": {
        "type": "Использование телефона при работе с наличными",
        "description": "Кассир пользуется мобильным телефоном (смотрит в экран, снимает, разговаривает), пока на столе наличные или идёт их пересчёт.",
        "hint": "the cashier actively uses a mobile phone (looks at its screen, takes photos, talks) while cash lies on the desk or is being counted; just holding it or putting it into a pocket is NOT a violation",
        "rule": {"document": R231, "clause": "п. 12"},
    },
    "cash_from_client_stack": {
        "type": "Выдача купюр из пачки клиента",
        "description": "После пересчёта кассир вынимает купюры из принятой у клиента пачки и передаёт их клиенту.",
        "hint": "right after counting the customer's stack, the cashier takes banknotes out of it and hands them back",
        "rule": {"document": R231, "clause": "п. 20, п. 31"},
    },
}


# Тип для нарушений, которых нет в списке (ТЗ, FAQ 9: в тесте могут быть новые типы).
# Пункт не указываем, чтобы не выдумывать: ссылка на документ без пункта допустима (FAQ 31).
RULES["other"] = {
    "type": "Другое нарушение кассовой дисциплины",
    "description": "Замечено действие, которое может нарушать порядок работы с наличными или ценностями, но не входит в список известных типов. Требуется проверка человеком.",
    "hint": "a CLEAR cash-handling or security violation that is NOT in the list above (describe exactly what you see); use only if you are sure and none of the keys above fits",
    "rule": {"document": R231, "clause": ""},
}
OTHER_KEY = "other"

# --- совместимость с ML-движком (app/vlm_pipeline.py): ключ — id нарушения, title — название, hint — подсказка для модели ---
for _k, _r in RULES.items():
    _r.setdefault("title", _r["type"])

# НЕ нарушения по разметке организаторов — сообщаем модели явно, чтобы не было ложных срабатываний
NOT_VIOLATIONS = [
    "the customer gives money without showing an ID document",
    "the cashier holds a mobile phone or puts it into a pocket without using it during cash operations",
    "the room door is closed but not locked",
    "the laptop screen is lit but no program window is open",
    "the light is left on",
    "the cashier works alone, without a controller",
    "cash lies on the desk while the cashier is actively counting or handling it",
    "checked banknotes are put together into one stack",
]


def rules_for_prompt() -> str:
    """Список правил для промпта модели: '- key: hint' + что нарушением НЕ является."""
    lines = [f"- {key}: {r['hint']}" for key, r in RULES.items() if r.get("hint")]
    lines.append("NOT violations (never report these): " + "; ".join(NOT_VIOLATIONS) + ".")
    return "\n".join(lines)


def rule_text(rule):
    """«Документ, пункт» одной строкой."""
    if not rule or not rule.get("document"):
        return ""
    return rule["document"] + (", " + rule["clause"] if rule.get("clause") else "")
