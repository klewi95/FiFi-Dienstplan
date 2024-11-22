import streamlit as st
import pandas as pd
import pulp
from datetime import datetime, timedelta
import holidays
import json
import os
import matplotlib.pyplot as plt
import csv
import io

# Seitenkonfiguration
st.set_page_config(page_title="Dienstplan-Manager", layout="wide")

# Datei für Mitarbeiterdaten
EMPLOYEE_FILE = 'employees.json'

# Funktion zum Laden der Mitarbeiterdaten
def load_employees():
    if not os.path.exists(EMPLOYEE_FILE):
        return {}
    with open(EMPLOYEE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

# Funktion zum Speichern der Mitarbeiterdaten
def save_employees(employees):
    with open(EMPLOYEE_FILE, 'w', encoding='utf-8') as f:
        json.dump(employees, f, ensure_ascii=False, indent=4)

# Laden der Mitarbeiterdaten beim Start der Anwendung
employees = load_employees()

# Definition der Schichten und deren Dauer sowie Startzeiten
shifts_weekday = {
    'Early': {'duration': 8, 'start': 6.75},    # 6:45 Uhr
    'Late': {'duration': 8, 'start': 14.75}     # 14:45 Uhr
}

shifts_weekend = {
    'Early': {'duration': 5, 'start': 9.25},    # 9:15 Uhr
    'Late': {'duration': 6, 'start': 14.25}     # 14:15 Uhr
}

# Betriebsratvorschriften und Arbeitszeitgesetz
max_consecutive_days = 6  # Maximal 6 aufeinanderfolgende Arbeitstage
max_daily_hours = 8       # Maximal 8 Stunden pro Tag
min_rest_time = 11        # Mindestruhezeit zwischen Schichten in Stunden

# Parameter für Fairness und Soft Constraints
allowed_shift_deviation = 2  # Erlaubte Abweichung von der durchschnittlichen Schichtanzahl
penalty_per_day = 100        # Strafwert für jeden Tag über dem Soft-Limit

# Zeitraum für den Dienstplan
start_date = datetime(2024, 10, 1)
end_date = datetime(2024, 10, 31)
dates = pd.date_range(start_date, end_date)

# Feiertage in NRW 2024 automatisch berechnen
deutschland_feiertage = holidays.Germany(years=2024, prov='NW')  # 'NW' für Nordrhein-Westfalen
feiertage = set([date for date in dates if date in deutschland_feiertage])

# Hilfsfunktionen zur Bestimmung von Wochentagen und Schichtinformationen
def is_weekend_or_holiday(date):
    return date.weekday() >= 5 or date in feiertage

def get_shift_duration(shift, date):
    is_weekend = is_weekend_or_holiday(date)
    if is_weekend:
        return shifts_weekend[shift]['duration']
    else:
        return shifts_weekday[shift]['duration']

def get_shift_start(shift, date):
    is_weekend = is_weekend_or_holiday(date)
    if is_weekend:
        return shifts_weekend[shift]['start']
    else:
        return shifts_weekday[shift]['start']

def get_actual_working_time(shift, date):
    duration = get_shift_duration(shift, date)
    if duration > 6:
        return duration - 1  # Abzug der 1-stündigen Pause
    else:
        return duration  # Keine Pause erforderlich

def get_preference_score(employee, date, shift):
    preferences = employees[employee].get('preferences', {})
    date_str = date.strftime('%Y-%m-%d')
    day_name = date.strftime('%A')

    # Prüfe auf spezifische Datumsvorgabe
    date_pref = preferences.get(date_str)
    if date_pref is not None:
        return date_pref.get(shift, 0)

    # Prüfe auf Wochentagspräferenz
    day_pref = preferences.get(day_name)
    if day_pref is not None:
        return day_pref.get(shift, 0)

    # Standardpräferenz
    return 0

# Funktion zur Generierung des Dienstplans
def generate_schedule():
    if not employees:
        return None, "Keine Mitarbeiterdaten vorhanden."

    # Initialisieren des Optimierungsproblems
    prob = pulp.LpProblem("Dienstplan", pulp.LpMaximize)

    # Entscheidungsvariablen: Zuordnung von Mitarbeitern zu Schichten
    assignments = {}
    for employee in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            is_weekend = is_weekend_or_holiday(date)
            available_shifts = shifts_weekend.keys() if is_weekend else shifts_weekday.keys()
            for shift in available_shifts:
                var = pulp.LpVariable(f"{employee}_{date_str}_{shift}", cat='Binary')
                assignments[(employee, date_str, shift)] = var

    # Berechnung der durchschnittlichen Schichten pro Mitarbeiter
    total_shifts = len(dates) * 2  # Pro Tag 2 Schichten
    num_employees = len(employees)
    total_max_hours = sum([employees[e]['max_weekly_hours'] for e in employees])
    average_shifts_per_employee = {e: (employees[e]['max_weekly_hours'] / total_max_hours) * total_shifts for e in employees}

    # Gewichtungen für die Zielfunktion
    preference_weight = 10  # Gewichtung der Präferenzen
    penalty_weight = 1      # Gewichtung der Strafwerte

    # Berechnung der Präferenzpunkte
    preference_score = pulp.lpSum([
        assignments.get((e, d.strftime('%Y-%m-%d'), s), 0) * get_preference_score(e, d, s)
        for e in employees
        for d in dates
        for s in ['Early', 'Late']
        if (e, d.strftime('%Y-%m-%d'), s) in assignments
    ])

    # Strafwerte für Soft Constraints (maximale aufeinanderfolgende Arbeitstage)
    penalty_terms = []
    consecutive_days_vars = {}
    for e in employees:
        for idx in range(len(dates) - max_consecutive_days):
            var = pulp.LpVariable(f"ConsecutiveDays_{e}_{idx}", lowBound=0, cat='Integer')
            consecutive_days_vars[(e, idx)] = var
            # Berechnung der aufeinanderfolgenden Arbeitstage
            total_consecutive = pulp.lpSum([
                pulp.lpSum([assignments.get((e, dates[idx + j].strftime('%Y-%m-%d'), shift), 0) for shift in ['Early', 'Late']])
                for j in range(max_consecutive_days + 1)
            ])
            prob += var >= total_consecutive - max_consecutive_days, f"SoftMaxConsecutiveDays_{e}_{idx}"
            # Hinzufügen zum Strafwert
            penalty_terms.append(var * penalty_per_day)

    # Zielfunktion: Maximierung der Präferenzen minus Strafwerte
    prob += preference_weight * preference_score - penalty_weight * pulp.lpSum(penalty_terms), "ObjectiveFunction"

    # Nebenbedingungen

    # 1. Personalbesetzung pro Schicht
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        is_weekend = is_weekend_or_holiday(date)
        available_shifts = shifts_weekend.keys() if is_weekend else shifts_weekday.keys()
        for shift in available_shifts:
            total_staff = pulp.lpSum([assignments.get((e, date_str, shift), 0) for e in employees])
            # Mindestpersonal
            prob += total_staff >= 2, f"MinStaff_{date_str}_{shift}"
            # Maximalpersonal
            prob += total_staff <= 3, f"MaxStaff_{date_str}_{shift}"

    # 2. Wöchentliche Arbeitszeit pro Mitarbeiter
    weeks = {}
    for date in dates:
        week_num = date.isocalendar()[1]
        if week_num not in weeks:
            weeks[week_num] = []
        weeks[week_num].append(date.strftime('%Y-%m-%d'))

    for e in employees:
        for week_num, week_dates in weeks.items():
            total_weekly_hours = pulp.lpSum([
                (assignments.get((e, d, 'Early'), 0) * get_actual_working_time('Early', pd.to_datetime(d))) +
                (assignments.get((e, d, 'Late'), 0) * get_actual_working_time('Late', pd.to_datetime(d)))
                for d in week_dates
            ])
            # Maximale wöchentliche Arbeitszeit
            prob += total_weekly_hours <= employees[e]['max_weekly_hours'], f"MaxWeeklyHours_{e}_Week_{week_num}"
            # Minimale wöchentliche Arbeitszeit
            prob += total_weekly_hours >= employees[e]['min_weekly_hours'], f"MinWeeklyHours_{e}_Week_{week_num}"

    # 3. Maximale tägliche Arbeitszeit
    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            daily_hours = pulp.lpSum([
                (assignments.get((e, date_str, 'Early'), 0) * get_actual_working_time('Early', date)) +
                (assignments.get((e, date_str, 'Late'), 0) * get_actual_working_time('Late', date))
            ])
            prob += daily_hours <= max_daily_hours, f"MaxDailyHours_{e}_{date_str}"

    # 4. Maximal eine Schicht pro Tag pro Mitarbeiter
    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            prob += pulp.lpSum([
                assignments.get((e, date_str, 'Early'), 0),
                assignments.get((e, date_str, 'Late'), 0)
            ]) <= 1, f"MaxOneShiftPerDay_{e}_{date_str}"

    # 5. Berücksichtigung der Verfügbarkeiten und Einschränkungen
    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            day_name = date.strftime('%A')
            available_shifts = employees[e]['availability'].get(day_name, [])
            # Einschränkungen für bestimmte Tage
            restricted_shifts = employees[e].get('restrictions', {}).get(date_str, [])
            for shift in ['Early', 'Late']:
                if shift not in available_shifts or shift in restricted_shifts:
                    if (e, date_str, shift) in assignments:
                        prob += assignments[(e, date_str, shift)] == 0, f"Restriction_{e}_{date_str}_{shift}"

    # 6. Maximal erlaubte aufeinanderfolgende Arbeitstage (harte Grenze)
    for e in employees:
        for idx in range(len(dates) - max_consecutive_days):
            total_consecutive = pulp.lpSum([
                pulp.lpSum([assignments.get((e, dates[idx + j].strftime('%Y-%m-%d'), shift), 0) for shift in ['Early', 'Late']])
                for j in range(max_consecutive_days + 1)
            ])
            prob += total_consecutive <= max_consecutive_days, f"MaxConsecutiveDays_{e}_{idx}"

    # 7. Mindestruhezeit zwischen Schichten
    for e in employees:
        for idx in range(len(dates) - 1):
            current_date = dates[idx]
            next_date = dates[idx + 1]
            current_date_str = current_date.strftime('%Y-%m-%d')
            next_date_str = next_date.strftime('%Y-%m-%d')
            for current_shift in ['Early', 'Late']:
                for next_shift in ['Early', 'Late']:
                    if (e, current_date_str, current_shift) in assignments and (e, next_date_str, next_shift) in assignments:
                        # Endzeit der aktuellen Schicht
                        end_time_current = get_shift_start(current_shift, current_date) + get_shift_duration(current_shift, current_date)
                        # Startzeit der nächsten Schicht
                        start_time_next = get_shift_start(next_shift, next_date)
                        # Berechnung der Ruhezeit
                        rest_time = (start_time_next + (24 if start_time_next <= end_time_current else 0)) - end_time_current
                        if rest_time < min_rest_time:
                            prob += assignments[(e, current_date_str, current_shift)] + assignments[(e, next_date_str, next_shift)] <= 1, f"MinRestTime_{e}_{current_date_str}_{current_shift}_{next_date_str}_{next_shift}"

    # 8. Fairness in der Schichtverteilung
    for e in employees:
        total_shifts_assigned = pulp.lpSum([
            assignments.get((e, d.strftime('%Y-%m-%d'), shift), 0)
            for d in dates
            for shift in ['Early', 'Late']
        ])
        # Fairnessbedingungen
        prob += total_shifts_assigned >= average_shifts_per_employee[e] - allowed_shift_deviation, f"FairnessMin_{e}"
        prob += total_shifts_assigned <= average_shifts_per_employee[e] + allowed_shift_deviation, f"FairnessMax_{e}"

    # 9. Fairness bei Wochenend- und Feiertagsschichten
    def calculate_weekend_holiday_shifts(employee):
        return pulp.lpSum([
            assignments.get((employee, d.strftime('%Y-%m-%d'), s), 0)
            for d in dates
            for s in ['Early', 'Late']
            if is_weekend_or_holiday(d)
        ])

    total_weekend_holiday_shifts = pulp.lpSum([
        calculate_weekend_holiday_shifts(e)
        for e in employees
    ])

    avg_weekend_holiday_shifts = total_weekend_holiday_shifts / num_employees if num_employees > 0 else 0

    for e in employees:
        total_weekend_shifts = calculate_weekend_holiday_shifts(e)
        prob += total_weekend_shifts <= avg_weekend_holiday_shifts + 1, f"FairWeekendHoliday_{e}_Max"
        prob += total_weekend_shifts >= avg_weekend_holiday_shifts - 1, f"FairWeekendHoliday_{e}_Min"

    # 10. Rolling Average von 48 Stunden pro Woche über 4 Wochen
    for e in employees:
        for idx in range(len(dates)):
            if idx >= 27:  # Betrachtung der letzten 4 Wochen (28 Tage)
                start_idx = idx - 27
                period_dates = dates[start_idx:idx+1]
                total_hours = pulp.lpSum([
                    (assignments.get((e, d.strftime('%Y-%m-%d'), 'Early'), 0) * get_actual_working_time('Early', d)) +
                    (assignments.get((e, d.strftime('%Y-%m-%d'), 'Late'), 0) * get_actual_working_time('Late', d))
                    for d in period_dates
                ])
                prob += total_hours <= 48 * 4, f"RollingAvg48h_{e}_{dates[idx].strftime('%Y-%m-%d')}"

    # Lösen des Optimierungsproblems
    prob.solve()

    # Überprüfen, ob eine optimale Lösung gefunden wurde
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, f"Keine optimale Lösung gefunden. Status: {pulp.LpStatus[prob.status]}"

    # Erstellen des Dienstplans basierend auf der Lösung
    dienstplan = {e: [] for e in employees}

    for e in employees:
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            day_name = date.strftime('%A')
            for shift in ['Early', 'Late']:
                if (e, date_str, shift) in assignments and pulp.value(assignments[(e, date_str, shift)]) == 1:
                    start_time = get_shift_start(shift, date)
                    end_time = start_time + get_shift_duration(shift, date)
                    # Umrechnung der Zeiten in Stunden und Minuten
                    start_hour = int(start_time)
                    start_min = int((start_time - start_hour) * 60)
                    end_hour = int(end_time)
                    end_min = int((end_time - end_hour) * 60)
                    # Anpassung der Endzeit bei Überlauf über 24 Stunden
                    if end_hour >= 24:
                        end_hour -= 24
                    start_str = f"{start_hour:02d}:{start_min:02d}"
                    end_str = f"{end_hour:02d}:{end_min:02d}"
                    # Berechnung der tatsächlichen Arbeitszeit
                    duration = get_shift_duration(shift, date)
                    if duration > 6:
                        pause = True
                        arbeitszeit = duration - 1  # Abzug der Pause
                    else:
                        pause = False
                        arbeitszeit = duration
                    # Hinzufügen zur Dienstplanliste
                    dienstplan[e].append({
                        'Datum': date_str,
                        'Wochentag': day_name,
                        'Schicht': shift,
                        'Startzeit': start_str,
                        'Endzeit': end_str,
                        'Arbeitszeit (Std.)': arbeitszeit,
                        'Pause (1 Std.)': 'Ja' if pause else 'Nein'
                    })

    return dienstplan, "Optimal"

# Funktion zum Anzeigen des Dienstplans
def display_schedule(dienstplan):
    st.header("Dienstplan anzeigen")
    if dienstplan:
        for name, shifts in dienstplan.items():
            st.subheader(f"Mitarbeiter: {name}")
            df = pd.DataFrame(shifts)
            st.dataframe(df)
    else:
        st.info("Kein Dienstplan gefunden.")

# Hauptfunktion der Streamlit-App
def main():
    st.title("Dienstplan-Manager für Fitnessstudio")

    menu = ["Startseite", "Mitarbeiter verwalten", "Dienstplan erstellen", "Dienstplan anzeigen"]
    choice = st.sidebar.selectbox("Menü", menu)

    if choice == "Startseite":
        st.subheader("Willkommen zum Dienstplan-Manager")
        st.write("Bitte wählen Sie eine Option aus dem Menü.")

    elif choice == "Mitarbeiter verwalten":
        st.header("Mitarbeiterverwaltung")

        if st.button("Mitarbeiterdaten aktualisieren"):
            global employees
            employees = load_employees()
            st.success("Mitarbeiterdaten wurden aktualisiert.")

        # Anzeige der Mitarbeiterliste
        if employees:
            for e, details in employees.items():
                st.subheader(e)
                st.write(f"Max Wochenstunden: {details['max_weekly_hours']}")
                st.write(f"Min Wochenstunden: {details['min_weekly_hours']}")
                st.write("Verfügbarkeiten:")
                st.json(details.get('availability', {}))
                st.write("Einschränkungen:")
                st.json(details.get('restrictions', {}))
                st.write("Präferenzen:")
                st.json(details.get('preferences', {}))
        else:
            st.info("Keine Mitarbeiterdaten vorhanden.")

    elif choice == "Dienstplan erstellen":
        st.header("Dienstplan erstellen")
        if st.button("Dienstplan generieren"):
            with st.spinner('Dienstplan wird generiert...'):
                dienstplan, status = generate_schedule()
            if dienstplan:
                st.success("Dienstplan erfolgreich erstellt.")
                # Speichern des Dienstplans in Session State
                st.session_state['dienstplan'] = dienstplan
            else:
                st.error(f"Fehler bei der Dienstplanerstellung: {status}")

    elif choice == "Dienstplan anzeigen":
        dienstplan = st.session_state.get('dienstplan', None)
        if dienstplan:
            display_schedule(dienstplan)
            if st.button("Dienstplan als CSV herunterladen"):
                # Erstellen der CSV-Datei im Speicher
                output = io.StringIO()
                fieldnames = ['Mitarbeiter', 'Datum', 'Wochentag', 'Schicht', 'Startzeit', 'Endzeit', 'Arbeitszeit (Std.)', 'Pause (1 Std.)']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for e, shifts in dienstplan.items():
                    for s in shifts:
                        writer.writerow({
                            'Mitarbeiter': e,
                            'Datum': s['Datum'],
                            'Wochentag': s['Wochentag'],
                            'Schicht': s['Schicht'],
                            'Startzeit': s['Startzeit'],
                            'Endzeit': s['Endzeit'],
                            'Arbeitszeit (Std.)': s['Arbeitszeit (Std.)'],
                            'Pause (1 Std.)': s['Pause (1 Std.)']
                        })
                csv_data = output.getvalue()
                output.close()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name='dienstplan.csv',
                    mime='text/csv'
                )
        else:
            st.info("Kein Dienstplan gefunden. Bitte erstellen Sie zuerst einen Dienstplan.")

if __name__ == '__main__':
    main()
