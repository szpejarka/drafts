# AlphaZero Draughts Implementation

Educational implementation of an AlphaZero-style neural network for playing draughts (Polish checkers).

## Zasady Gry w Warcaby (Draughts Rules)

### Plansza i Początkowe Ustawienie
- **Rozmiar**: 8×8
- **Pola gry**: Tylko ciemne pola (64 pola do gry)
- **Początek**: Każdy gracz ma 12 pionów
  - Gracz 1 (biały) - rzędy 0-2
  - Gracz 2 (czerwony) - rzędy 5-7

### Rodzaje Figur
- **Pion (man)**: Porusza się tylko do przodu po przekątnej, o jedno pole. **Nie może się poruszać do tyłu.**
- **Dama (king)**: Porusza się po przekątnej w każdym kierunku (w przód i w tył), o jedno pole

### Poruszanie Się Piona
1. Pion może przesunąć się tylko **do przodu** na sąsiadujące puste pole po przekątnej
2. Pion **nigdy nie porusza się do tyłu** (nawet podczas normalnego ruchu)
3. Przy biciu (capture) pion przeskakuje figurę przeciwnika **do przodu** o dwa pola po przekątnej
4. Pion nie może bić do tyłu

### Poruszanie Się Damy
1. Dama może przesunąć się po przekątnej w **każdym kierunku** (do przodu i do tyłu) o jedno pole
2. Dama może bić figurę przeciwnika **w każdym kierunku** (do przodu i do tyłu), przeskakując o dwa pola po przekątnej
3. Dama nie jest ograniczona kierunkiem (w przeciwieństwie do piona)

### Bicie (Capture)
Jeśli figurę przeciwnika można przeskoczyć:
- Skacz o dwa pola po przekątnej
- Pole pośrednie musi zawierać figurę przeciwnika
- Pole docelowe musi być puste
- Zbitą figurę usuwam z planszy
- **Pion może bić tylko do przodu; dama może bić w każdym kierunku**

### Bicie Obowiązkowe (Forced Capture)
- Jeśli dostępne jest bicie, **musi** ono być wykonane
- Po biciu, jeśli figura może bić dalej, **musi** kontynuować bicie (multi-capture)
- Gracz nie może przerwać serii bić

### Awans na Damę
- Pion osiągający ostatni rząd (rząd 7 dla białego, rząd 0 dla czerwonego) awansuje do roli damy
- Po awansie figura zyskuje możliwość poruszania się i bicia w każdym kierunku

### Koniec Gry
- Gra kończy się, gdy gracz nie ma legalnych ruchów (przegrana)
- Remis może być ogłoszony po wielokrotnym powtórzeniu pozycji

## Kodowanie Figur
```
0: Puste pole
1: Pion gracza 1 (biały)
2: Dama gracza 1 (biały)
3: Pion gracza 2 (czerwony)
4: Dama gracza 2 (czerwony)
```

## Reprezentacja Sieciowa
Plansza jest konwertowana do tensora 5-kanałowego:
- Kanał 0: Piony gracza 1
- Kanał 1: Damy gracza 1
- Kanał 2: Piony gracza 2
- Kanał 3: Damy gracza 2
- Kanał 4: Ruch (0 = gracz 1, 1 = gracz 2)

## Użycie

```bash
python GameState.py
```

Gra uruchamia się interaktywnie - sieć neuronowa wybiera ruchy, a użytkownik może kontynuować lub przerwać (wpisując 'q').
