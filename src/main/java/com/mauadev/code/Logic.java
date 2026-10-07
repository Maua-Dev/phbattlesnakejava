package com.mauadev.code;

// =============================================================================
//  COMO ESTA COBRA PENSA  (v6: Java, baseada no logic.py v5.0.0 — leia antes de mexer no código)
// =============================================================================
//
//  A cada turno o jogo manda o estado do tabuleiro e a cobra tem 500 ms para responder
//  "up", "down", "left" ou "right". O caminho de getMove() é:
//
//   1. LER o tabuleiro ........ buildContext(): tamanho, regras (standard, royale, wrapped,
//                               constrictor), hazards, comida, cobras. Nada é fixo em 11x11.
//   2. TIRAR o suicídio ....... getPossibleMoves(): parede, corpos, casas que não liberam a tempo.
//   3. ESCOLHER entre o resto:
//        - 1 rival (1v1, o formato do campeonato) -> BUSCA minimax (classe Search, no fim deste arquivo).
//        - 3+ cobras -> PONTUAÇÃO por componentes (seção "COMPONENTES DE PONTUAÇÃO").
//   4. SE ALGO FALHAR ......... fallback seguro: nunca devolve erro, nem estoura o prazo.
//
//  REGRAS DO JOGO QUE O CÓDIGO RESPEITA
//   - Movimento simultâneo: as cobras escolhem ao mesmo tempo. Logo não dá para "prever" o
//     rival; planejamos para o PIOR caso dele (minimax).
//   - Cabeça contra cabeça: a MENOR morre; de mesmo tamanho morrem as duas. Casa que cobra
//     maior/igual alcança é perigosa; casa que só cobra menor alcança é uma chance de matar.
//   - Cauda: sai do lugar no turno seguinte, então pisar nela é seguro, EXCETO se a cobra
//     acabou de comer (cauda duplicada) ou pode comer agora.
//   - Vida: -1 por turno, comida devolve 100 e cresce 1; hazard custa mais 14 por turno
//     (royale). 'wrapped': as bordas dão a volta. 'constrictor': sem comida, todos crescem
//     todo turno e a cauda nunca libera casa.
//
//  ESTRATÉGIAS (por que ela joga assim)
//   - TERRITÓRIO (Voronoi): cada casa pertence a quem chega nela primeiro. Quem tem mais casas
//     tem mais comida, mais saída e acaba encurralando o outro. É o núcleo da avaliação.
//   - ENCURRALAR: a busca vê, a vários turnos de distância, jogadas que reduzem o território
//     do rival até ele não ter saída, e evita as que fazem isso comigo.
//   - TAMANHO: ser maior vence choques de cabeça e disputas de casa; por isso comer vale, mas
//     só quando é seguro (comida disputada por cobra maior é ignorada).
//   - FOME: se não dá para chegar à comida antes da vida acabar, a nota despenca. Com hazards
//     o cálculo é em PONTOS DE VIDA (cada casa de hazard custa 1 + dano), não em passos.
//   - SEGUIR A CAUDA: com o corpo grande, o caminho mais seguro costuma ser seguir a própria
//     cauda: ela sempre abre espaço.
//   - TEMPO: a jogada inteira termina em até 100 ms (MOVE_MAX_MS). A busca aprofunda de 1 em 1
//     (iterative deepening) e para quando o prazo acaba (SEARCH_CAP_MS), então sempre há resposta.
//     O prazo encolhe sozinho se a latência medida estiver alta.
//
//  TÉCNICAS (resumo)
//   - Minimax com poda alfa-beta, aprofundamento iterativo e ordenação pelas notas clássicas.
//   - Ordenação "killer": em cada nível da árvore, o lance que foi melhor da última vez é testado
//     primeiro. Não muda a resposta, mas a poda corta muito mais cedo (quase metade dos nós).
//   - Avaliação com BITBOARDS: o território (Voronoi) é calculado com operações de bits, várias
//     casas por instrução. Resultado idêntico ao cálculo casa por casa, ~5x mais rápido.
//   - Fim de jogo: vitória = +100000 - turno (quanto antes melhor); empate = 0; derrota = -(...).
//   - Tudo que mexe em comportamento está em Weights / constantes TUNING / Search.W_* (no fim do arquivo).
//
//  Documentação do jogo: https://docs.battlesnake.com

import com.mauadev.code.entities.Board;
import com.mauadev.code.entities.Coordinate;
import com.mauadev.code.entities.Game;
import com.mauadev.code.entities.GameState;
import com.mauadev.code.entities.Snake;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;

public class Logic {

    // --------------------------------------------------------------------------- //
    // CONFIGURAÇÃO (tudo que você vai querer ajustar fica aqui)
    // --------------------------------------------------------------------------- //

    static final String[] MOVE_ORDER = {"up", "down", "left", "right"};
    static final int[] DX = {0, 0, -1, 1};
    static final int[] DY = {1, -1, 0, 0};
    static final int INF = 1_000_000;
    static final int MAX_HEALTH = 100;

    /** Pesos da pontuação. Escala: "mortal" ~ 700-900, "importante" ~ 100-300, "desempate" ~ 10-40. */
    static final class Weights {
        double h2hLoss = 900.0;     // casa que uma cobra MAIOR pode alcançar: perderíamos o head-to-head
        double h2hTie = 700.0;      // idem com cobra de MESMO tamanho: as duas morrem
        double deadEnd = 700.0;     // penalidade máxima quando a região acessível é menor que o corpo
        double space = 200.0;       // espaço acessível (flood fill), saturando em um "conforto"
        double mobility = 20.0;     // por saída livre logo após o movimento
        double tailReach = 40.0;    // bônus por conseguir alcançar a própria cauda (rota de fuga)
        double territory = 250.0;   // fatia do tabuleiro que chegamos ANTES dos adversários
        double center = 25.0;       // leve preferência pelo centro
        double edge = 12.0;         // leve penalidade por borda
        double food = 300.0;        // valor da melhor comida alcançável (modulado por urgência)
        double starvation = 300.0;  // não chegaremos a nenhuma comida antes de morrer de fome
        double lowHealth = 60.0;    // penalidade crescente conforme a vida cai de LOW_HEALTH
        double hazard = 60.0;       // custo de entrar em hazard
        double kill = 250.0;        // chance de eliminar cobra MENOR via head-to-head
        double threat = 60.0;       // proximidade da cabeça de cobra maior/igual
        double hunt = 80.0;         // aproximação de cabeça de cobra menor (quando saudável)
        double tailFollow = 120.0;  // seguir a própria cauda quando o espaço está apertado

        /** Aplica os multiplicadores da fase da partida (só o que for listado muda). */
        Weights forPhase(String phase) {
            Weights w = new Weights();
            if (phase.equals("opening")) {
                w.food = food * 1.3;
                w.hunt = hunt * 0.3;
                w.kill = kill * 0.7;
            } else if (phase.equals("endgame")) {   // 1v1: pressionar o adversário
                w.hunt = hunt * 2.0;
                w.territory = territory * 1.3;
                w.food = food * 0.8;
            }
            return w;
        }
    }

    static final Weights BASE = new Weights();

    // Limiares e fatores (não são "pesos", mas também são ajustáveis).
    static final double HEALTH_CRITICAL = 25;       // abaixo disso, urgência de comida = 1.0
    static final double HEALTH_COMFORT = 65;        // acima disso, urgência = 0.0
    static final double HAZARD_URGENCY_SHIFT = 15;  // em mapas com hazard a vida se esgota mais rápido
    static final double FOOD_BASE = 0.25;           // interesse mínimo em comida (crescer) mesmo com vida cheia
    static final double FOOD_BASE_AHEAD = 0.10;     // idem quando já somos bem maiores que todos
    static final int SCARCE_FOOD = 2;               // com <= N comidas no mapa, comida vale um pouco mais
    static final double CONTESTED = 0.2;            // fator se um adversário chega antes
    static final double HAZARD_FOOD = 0.4;          // fator para comida dentro de hazard (se não urgente)
    static final double DEADEND_FOOD = 0.4;         // fator para comida em beco (<= 1 saída)
    static final int HAZARD_DAMAGE = 14;            // dano padrão do hazard por turno (royale)
    static final double LOW_HEALTH = 20;            // abaixo disso começamos a penalizar vida baixa
    static final double THREAT_RANGE = 3;           // distância em que cabeças maiores nos assustam
    static final double HUNT_RANGE = 4;             // distância em que caçamos cabeças menores
    static final int SPACE_COMFORT_MIN = 20;        // espaço "confortável" mínimo
    static final int TAIL_MIN_LEN = 8;              // só seguimos a cauda ativamente com corpo desse tamanho
    static final int OPENING_TURNS = 12;            // turnos considerados "abertura"
    static final double TIME_BUDGET = 0.4;          // fração do timeout que podemos gastar calculando
    static final double SEARCH_TIME = 0.30;         // fração do timeout usada pela busca 1v1
    static final double SEARCH_CAP_MS = 65.0;       // teto da busca 1v1 (ms)
    // TEMPO MÁXIMO DA JOGADA: tudo (ler o estado, buscar, desempatar) termina antes disso.
    static final double MOVE_MAX_MS = 100.0;
    static final double TIEBREAK_UNTIL_MS = 78.0;   // depois disso o desempate usa a nota rápida
    static final double CLASSIC_UNTIL_MS = 85.0;    // 3+ cobras: depois disso só a nota rápida

    /** Orçamento fixo da busca em ms (testes). Negativo = automático (ver searchBudgetMs). */
    static double searchMsOverride = Double.parseDouble(System.getProperty("battlesnake.searchMs", "-1"));
    /** Escreve uma linha por jogada na saída padrão (CloudWatch na Lambda). */
    static boolean log = false;
    /** Duração do aquecimento feito em start(). Curto de propósito: o /start também tem prazo. */
    static final long WARMUP_MS = 100;

    /** Só roda o aquecimento uma vez por instância da Lambda. */
    private static boolean aquecida = false;

    // --------------------------------------------------------------------------- //
    // INFO / START / END
    // --------------------------------------------------------------------------- //

    /**
     * GET / - aparência da cobra. Opções de cabeça, cauda e cor:
     * https://docs.battlesnake.com/guides/customizations
     */
    public static Map<String, String> info() {
        Map<String, String> info = new HashMap<>();
        info.put("apiversion", "1");
        info.put("author", "ph-ARCH");          // TODO: coloque aqui o SEU usuário do Battlesnake
        info.put("color", "#8b0051");    // TODO: escolha a cor da sua cobra
        info.put("head", "Snowman");  // TODO: escolha a cabeça
        info.put("tail", "Mouse");        // TODO: escolha a cauda
        info.put("version", "6.0.0-java");
        return info;
    }

    /**
     * POST /start - uma vez por partida. Aproveitamos para "esquentar" a JVM: nas primeiras jogadas de
     * uma instância nova da Lambda o código Java ainda roda sem otimização (a busca ficaria mais rasa).
     */
    public static void start(GameState state) {
        if (!aquecida) {
            aquecida = true;
            try {
                warmUp(WARMUP_MS);
            } catch (RuntimeException e) {
                // aquecimento é só um extra: nunca pode atrapalhar a partida
            }
        }
    }

    /** POST /end - uma vez por partida, quando ela termina. */
    public static void end(GameState state) {
        // Nada a fazer: a cobra não guarda estado entre as partidas.
    }

    // --------------------------------------------------------------------------- //
    // MODELOS INTERNOS
    // --------------------------------------------------------------------------- //
    // Casas são números: cell = y * largura + x.

    static final class Enemy {
        final String id;
        final int head;
        final int[] body;
        final int length;
        final int health;
        int[] nextCells = new int[0];   // casas que a cabeça pode ocupar no próximo turno

        Enemy(String id, int[] body, int health) {
            this.id = id;
            this.head = body[0];
            this.body = body;
            this.length = body.length;
            this.health = health;
        }
    }

    static final class Ctx {
        int width, height, V, turn;
        boolean wrapped, constrictor;
        int hazardDamage;
        int myHead, myTail, myLen, myHealth;
        int[] myBody;
        List<Enemy> enemies = new ArrayList<>();
        int[] food;                  // lista de comidas
        boolean[] isFood, isHazard;
        boolean anyHazard;
        int[] freeAt;                // casa -> turno (após o movimento) em que deixa de estar ocupada (0 = livre)
        long deadline;               // System.nanoTime()
        String phase = "midgame";
        double urgency;
        Weights w;
        int[] arrT, arrLen;          // enemy_arrival: turno de chegada (-1 = ninguém chega) e tamanho
        int maxEnemyLen;
        int[][] nbr;                 // vizinhos na ordem MOVE_ORDER; -1 = parede
    }

    static final class Flood {
        int[] arrival;               // casa -> turno em que chegamos (-1 = não chegamos)
        int count;
        boolean tailReachable;
    }

    // --------------------------------------------------------------------------- //
    // LEITURA DO STATE E CONSTRUÇÃO DO CONTEXTO
    // --------------------------------------------------------------------------- //

    static int[][] buildNbr(int W, int H, boolean wrapped) {
        int V = W * H;
        int[][] nbr = new int[V][4];
        for (int c = 0; c < V; c++) {
            int x = c % W, y = c / W;
            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if (wrapped) {
                    nx = ((nx % W) + W) % W;
                    ny = ((ny % H) + H) % H;
                    nbr[c][d] = ny * W + nx;
                } else if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    nbr[c][d] = ny * W + nx;
                } else {
                    nbr[c][d] = -1;
                }
            }
        }
        return nbr;
    }

    /**
     * Converte coordenadas (x, y) em números de casa. Aceita lista nula. Se algum ponto cair fora do
     * tabuleiro o estado é inválido: lança exceção e getMove() cai no fallback seguro (legacySafeMove).
     */
    private static int[] cells(List<Coordinate> cs, int W, int H) {
        if (cs == null) return new int[0];
        int[] r = new int[cs.size()];
        for (int i = 0; i < r.length; i++) {
            Coordinate p = cs.get(i);
            if (p == null || p.getX() < 0 || p.getX() >= W || p.getY() < 0 || p.getY() >= H) {
                throw new IllegalArgumentException("coordenada fora do tabuleiro");
            }
            r[i] = p.getY() * W + p.getX();
        }
        return r;
    }

    /** Corpo da cobra, da cabeça até a cauda. Se só veio 'head', o corpo é essa única casa. */
    private static List<Coordinate> bodyOf(Snake s) {
        List<Coordinate> body = s.getBody();
        if (body != null && !body.isEmpty()) return body;
        return s.getHead() == null ? Collections.<Coordinate>emptyList() : Collections.singletonList(s.getHead());
    }

    // ------------------------------------------------------------------ MODO DE JOGO
    // As classes de entities/ (que o template pede para NÃO alterar) não trazem o "ruleset".
    // Então a cobra descobre o modo olhando o próprio tabuleiro:

    /** Todas as cobras do tabuleiro (inclui 'you' mesmo se ele não vier na lista). */
    private static List<Snake> allSnakes(GameState state) {
        List<Snake> all = new ArrayList<>();
        if (state.getBoard() != null && state.getBoard().getSnakes() != null) {
            for (Snake s : state.getBoard().getSnakes()) if (s != null) all.add(s);
        }
        Snake you = state.getYou();
        boolean listed = false;
        for (Snake s : all) if (s == you || (s.getId() != null && s.getId().equals(you.getId()))) listed = true;
        if (!listed && you != null) all.add(you);
        return all;
    }

    /**
     * WRAPPED (bordas dão a volta): algum corpo tem dois segmentos seguidos que não são vizinhos
     * no tabuleiro normal, ex. (0,5) -> (10,5). Antes de alguém atravessar a borda não dá para saber;
     * aí tratamos a borda como parede, o que só é mais cauteloso.
     */
    static boolean detectWrapped(GameState state) {
        for (Snake s : allSnakes(state)) {
            List<Coordinate> b = bodyOf(s);
            for (int i = 1; i < b.size(); i++) {
                Coordinate p = b.get(i - 1), q = b.get(i);
                if (p == null || q == null) continue;
                if (Math.abs(p.getX() - q.getX()) + Math.abs(p.getY() - q.getY()) > 1) return true;
            }
        }
        return false;
    }

    /**
     * CONSTRICTOR (todos crescem todo turno, sem comida): não há comida no tabuleiro (no standard
     * e no royale sempre há pelo menos 1), e todas as cobras estão com vida 100 e cauda duplicada.
     */
    static boolean detectConstrictor(GameState state) {
        Board b = state.getBoard();
        if (b == null || (b.getFood() != null && !b.getFood().isEmpty())) return false;
        for (Snake s : allSnakes(state)) {
            List<Coordinate> body = bodyOf(s);
            if (s.getHealth() < 100 || body.size() < 2) return false;
            Coordinate t1 = body.get(body.size() - 1), t2 = body.get(body.size() - 2);
            if (t1 == null || t2 == null || t1.getX() != t2.getX() || t1.getY() != t2.getY()) return false;
        }
        return true;
    }

    private static int timeoutOf(GameState state) {
        Game g = state.getGame();
        return g != null && g.getTimeout() > 0 ? g.getTimeout() : 500;
    }

    static Ctx buildContext(GameState state, long started) {
        Board board = state.getBoard();
        Snake you = state.getYou();
        Ctx c = new Ctx();
        c.width = board.getWidth();
        c.height = board.getHeight();
        c.V = c.width * c.height;
        c.turn = state.getTurn();

        c.constrictor = detectConstrictor(state);   // cauda nunca sai do lugar, sem comida
        c.wrapped = detectWrapped(state);           // bordas "dão a volta"
        c.hazardDamage = HAZARD_DAMAGE;             // royale: os hazards vêm em board.hazards
        int timeoutMs = timeoutOf(state);
        c.deadline = started + (long) (Math.min(timeoutMs * TIME_BUDGET, CLASSIC_UNTIL_MS) * 1_000_000.0);
        c.nbr = buildNbr(c.width, c.height, c.wrapped);

        c.myBody = cells(bodyOf(you), c.width, c.height);
        c.myHead = c.myBody[0];
        c.myTail = c.myBody[c.myBody.length - 1];
        c.myLen = c.myBody.length;
        c.myHealth = you.getHealth();

        List<int[]> bodies = new ArrayList<>();
        bodies.add(c.myBody);
        List<Snake> snakes = board.getSnakes() == null ? Collections.<Snake>emptyList() : board.getSnakes();
        for (Snake s : snakes) {
            if (s == null || Objects.equals(s.getId(), you.getId())) continue;   // 'you' também vem na lista
            int[] body = cells(bodyOf(s), c.width, c.height);
            if (body.length == 0) continue;
            bodies.add(body);
            c.enemies.add(new Enemy(s.getId(), body, s.getHealth()));
        }

        c.food = cells(board.getFood(), c.width, c.height);
        c.isFood = new boolean[c.V];
        for (int f : c.food) c.isFood[f] = true;
        c.isHazard = new boolean[c.V];
        int[] hz = cells(board.getHazards(), c.width, c.height);
        for (int h : hz) c.isHazard[h] = true;
        c.anyHazard = hz.length > 0;

        c.freeAt = buildFreeAt(c.V, bodies, c.constrictor);
        int maxLen = 0;
        for (Enemy e : c.enemies) maxLen = Math.max(maxLen, e.length);
        c.maxEnemyLen = maxLen;
        c.phase = getPhase(c);
        c.w = BASE.forPhase(c.phase);
        c.urgency = healthUrgency(c);

        // Rival colado numa comida pode comer neste turno: a cauda dele NÃO sai do lugar.
        // Tratamos a casa da cauda como ocupada no turno 1 (cuidado conservador).
        if (!c.constrictor) {
            for (Enemy e : c.enemies) {
                if (e.length > 1) {
                    boolean nearFood = false;
                    for (int n : c.nbr[e.head]) if (n >= 0 && c.isFood[n]) nearFood = true;
                    if (nearFood) {
                        int tail = e.body[e.body.length - 1];
                        c.freeAt[tail] = Math.max(c.freeAt[tail], 2);
                    }
                }
            }
        }
        for (Enemy e : c.enemies) {
            int cnt = 0;
            int[] tmp = new int[4];
            for (int n : c.nbr[e.head]) if (n >= 0 && c.freeAt[n] <= 1) tmp[cnt++] = n;
            e.nextCells = java.util.Arrays.copyOf(tmp, cnt);
        }
        buildEnemyArrival(c);
        return c;
    }

    /**
     * Para cada casa ocupada: em que turno (contado após o NOSSO movimento = 1) ela fica livre.
     * O segmento i de uma cobra de tamanho L sai do tabuleiro quando 'L - i' movimentos acontecem.
     * Logo a cauda (i = L-1) libera no turno 1: pisar nela é seguro. Se a cobra acabou de comer,
     * a cauda está duplicada; como pegamos o MAIOR valor entre segmentos sobrepostos, essa casa só
     * libera no turno 2 — automático. Em 'constrictor' ninguém libera casa nenhuma.
     */
    static int[] buildFreeAt(int V, List<int[]> bodies, boolean constrictor) {
        int[] freeAt = new int[V];
        for (int[] body : bodies) {
            int length = body.length;
            for (int i = 0; i < length; i++) {
                int t = constrictor ? INF : length - i;
                if (t > freeAt[body[i]]) freeAt[body[i]] = t;
            }
        }
        return freeAt;
    }

    /** Menor turno em que algum adversário chega a cada casa (usado em comida e território). */
    static void buildEnemyArrival(Ctx c) {
        c.arrT = new int[c.V];
        c.arrLen = new int[c.V];
        java.util.Arrays.fill(c.arrT, -1);
        for (Enemy e : c.enemies) {
            int[] arr = bfs(c, e.head, 0);
            for (int cell = 0; cell < c.V; cell++) {
                int t = arr[cell];
                if (t < 0) continue;
                if (c.arrT[cell] < 0 || t < c.arrT[cell] || (t == c.arrT[cell] && e.length > c.arrLen[cell])) {
                    c.arrT[cell] = t;
                    c.arrLen[cell] = e.length;
                }
            }
        }
    }

    static String getPhase(Ctx c) {
        if (c.turn < OPENING_TURNS) return "opening";
        if (c.enemies.size() == 1) return "endgame";
        return "midgame";
    }

    // --------------------------------------------------------------------------- //
    // GEOMETRIA
    // --------------------------------------------------------------------------- //

    static int distance(Ctx c, int a, int b) {
        int dx = Math.abs(a % c.width - b % c.width), dy = Math.abs(a / c.width - b / c.width);
        if (c.wrapped) {
            dx = Math.min(dx, c.width - dx);
            dy = Math.min(dy, c.height - dy);
        }
        return dx + dy;
    }

    static int openNeighbors(Ctx c, int pos, int turn) {
        int n = 0;
        for (int nb : c.nbr[pos]) if (nb >= 0 && c.freeAt[nb] <= turn) n++;
        return n;
    }

    // --------------------------------------------------------------------------- //
    // MOVIMENTOS POSSÍVEIS / SEGURANÇA
    // --------------------------------------------------------------------------- //

    /** Vida após entrar em 'pos'. Comer devolve vida cheia (e não custa vida). */
    static int healthAfterMove(Ctx c, int pos) {
        if (c.isFood[pos] && !c.constrictor) return MAX_HEALTH;
        int health = c.myHealth - 1;
        if (c.isHazard[pos]) health -= c.hazardDamage;
        return health;
    }

    /** Casa legal: dentro do tabuleiro, livre no turno 1 e sem morte por fome/hazard. */
    static boolean isPositionSafe(Ctx c, int pos) {
        if (pos < 0) return false;
        if (c.freeAt[pos] > 1) return false;
        return healthAfterMove(c, pos) > 0;
    }

    /** Movimentos (índices de MOVE_ORDER) que passam nos filtros de segurança. */
    static int[] getPossibleMoves(Ctx c) {
        int[] tmp = new int[4];
        int n = 0;
        for (int d = 0; d < 4; d++) if (isPositionSafe(c, c.nbr[c.myHead][d])) tmp[n++] = d;
        return java.util.Arrays.copyOf(tmp, n);
    }

    /**
     * Nenhum movimento passa nos filtros. Escolhe o "menos pior", de forma determinística:
     * 1) dentro do tabuleiro, 2) sobrevive à vida, 3) casa que libera mais cedo.
     */
    static int emergencyMove(Ctx c) {
        int best = 0, bestA = Integer.MAX_VALUE, bestB = Integer.MAX_VALUE;
        for (int d = 0; d < 4; d++) {
            int pos = c.nbr[c.myHead][d];
            int a, b;
            if (pos < 0) {
                a = 3;
                b = INF;
            } else {
                a = healthAfterMove(c, pos) > 0 ? 0 : 1;
                b = c.freeAt[pos];
            }
            if (a < bestA || (a == bestA && b < bestB)) {
                best = d;
                bestA = a;
                bestB = b;
            }
        }
        return best;
    }

    // --------------------------------------------------------------------------- //
    // BFS (turnos) e DIJKSTRA (custo em vida), ambos com liberação de caudas no tempo
    // --------------------------------------------------------------------------- //

    /** BFS: casa -> turno de chegada (-1 = não chega). Só atravessa casa que já esteja livre nesse turno. */
    static int[] bfs(Ctx c, int start, int startTurn) {
        int[] arrival = new int[c.V];
        java.util.Arrays.fill(arrival, -1);
        arrival[start] = startTurn;
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(start);
        while (!queue.isEmpty()) {
            int cell = queue.poll();
            int t = arrival[cell] + 1;
            for (int nxt : c.nbr[cell]) {
                if (nxt < 0 || arrival[nxt] >= 0 || c.freeAt[nxt] > t) continue;
                arrival[nxt] = t;
                queue.add(nxt);
            }
        }
        return arrival;
    }

    /** Espaço acessível a partir de 'start' (nossa cabeça após o movimento = turno 1). */
    static Flood floodFill(Ctx c, int start, int startTurn) {
        Flood f = new Flood();
        f.arrival = bfs(c, start, startTurn);
        int n = 0;
        for (int t : f.arrival) if (t >= 0) n++;
        f.count = n;
        f.tailReachable = !c.constrictor && c.myLen >= 2 && f.arrival[c.myTail] >= 0;
        return f;
    }

    /** Vida gasta ao ENTRAR numa casa: 1 por turno + dano extra se for hazard. */
    static int stepCost(Ctx c, int cell) {
        return 1 + (c.isHazard[cell] ? c.hazardDamage : 0);
    }

    static final class Routes {
        final int[] cost;   // custo em vida (INF = inalcançável)
        final int[] turn;   // turno de chegada pelo caminho escolhido

        Routes(int V) {
            cost = new int[V];
            turn = new int[V];
            java.util.Arrays.fill(cost, INF);
        }
    }

    private static long key(int cost, int turn, int cell) {
        return ((long) cost << 40) | ((long) turn << 20) | cell;
    }

    /**
     * Dijkstra a partir de 'start': casa -> (custo em vida, turno de chegada). O custo é medido em
     * PONTOS DE VIDA, então dá para comparar direto com a saúde (um caminho por hazard custa muito
     * mais do que o número de passos). O turno acompanha o caminho escolhido e serve para respeitar
     * a liberação das caudas.
     */
    static Routes dijkstraRoutes(Ctx c, int start, int startTurn) {
        Routes r = new Routes(c.V);
        r.cost[start] = 0;
        r.turn[start] = startTurn;
        PriorityQueue<Long> heap = new PriorityQueue<>();
        heap.add(key(0, startTurn, start));
        while (!heap.isEmpty()) {
            long k = heap.poll();
            int cost = (int) (k >>> 40), turn = (int) ((k >>> 20) & 0xFFFFF), cell = (int) (k & 0xFFFFF);
            if (r.cost[cell] != cost || r.turn[cell] != turn) continue;   // entrada obsoleta
            int t = turn + 1;
            for (int nxt : c.nbr[cell]) {
                if (nxt < 0 || c.freeAt[nxt] > t) continue;
                int cc = cost + stepCost(c, nxt);
                if (r.cost[nxt] == INF || cc < r.cost[nxt] || (cc == r.cost[nxt] && t < r.turn[nxt])) {
                    r.cost[nxt] = cc;
                    r.turn[nxt] = t;
                    heap.add(key(cc, t, nxt));
                }
            }
        }
        return r;
    }

    // --------------------------------------------------------------------------- //
    // COMPONENTES DE PONTUAÇÃO (cada um devolve pontos: positivo = bom)
    // --------------------------------------------------------------------------- //

    /** Espaço acessível + penalidade de beco (região menor que o corpo). */
    static double evaluateSpace(Ctx c, Flood f) {
        int comfort = Math.max(2 * c.myLen, SPACE_COMFORT_MIN);
        double score = c.w.space * Math.min(1.0, (double) f.count / comfort);
        if (f.count < c.myLen) {
            double penalty = c.w.deadEnd * (1.0 - (double) f.count / c.myLen);
            if (f.tailReachable) penalty *= 0.5;   // a cauda vai abrindo espaço: risco menor
            score -= penalty;
        }
        return score;
    }

    /** Fatia do tabuleiro que alcançamos antes (ou empatando sendo maiores) dos adversários. */
    static double evaluateTerritory(Ctx c, Flood f) {
        int mine = 0;
        for (int cell = 0; cell < c.V; cell++) {
            int t = f.arrival[cell];
            if (t < 0) continue;
            int et = c.arrT[cell];
            if (et < 0 || t < et || (t == et && c.myLen > c.arrLen[cell])) mine++;
        }
        return c.w.territory * mine / (c.width * c.height);
    }

    /** Quantas saídas existirão a partir da nova cabeça (turno 2). */
    static double evaluateMobility(Ctx c, int pos) {
        return c.w.mobility * openNeighbors(c, pos, 2);
    }

    /** Cauda alcançável = rota de fuga. Se o espaço está apertado e não há fome, seguir a cauda. */
    static double evaluateTail(Ctx c, Flood f) {
        if (!f.tailReachable) return 0.0;
        double score = c.w.tailReach;
        int comfort = Math.max(2 * c.myLen, SPACE_COMFORT_MIN);
        double tightness = Math.max(0.0, 1.0 - f.count / (1.5 * comfort));   // 0 = folgado, 1 = apertado
        if (c.myLen >= TAIL_MIN_LEN || tightness > 0.5) {
            int d = f.arrival[c.myTail] - 1;
            score += c.w.tailFollow * tightness * (1.0 - c.urgency) / (1 + d);
        }
        return score;
    }

    /** 0 = vida confortável, 1 = crítica (interpolação linear entre os dois limiares). */
    static double healthUrgency(Ctx c) {
        double lo = HEALTH_CRITICAL, hi = HEALTH_COMFORT;
        if (c.anyHazard) {
            lo += HAZARD_URGENCY_SHIFT;
            hi += HAZARD_URGENCY_SHIFT;
        }
        double h = c.myHealth;
        if (h <= lo) return 1.0;
        if (h >= hi) return 0.0;
        return (hi - h) / (hi - lo);
    }

    /**
     * Valor da MELHOR comida (não da mais próxima) a partir da nova posição. A distância vem do
     * Dijkstra (custo em vida: hazard pesa). Cada comida é descontada se: perderemos a corrida
     * (comparada em TURNOS), está em hazard (sem urgência) ou fica em beco. O interesse base sobe
     * com a urgência de vida.
     */
    static double evaluateFood(Ctx c, Routes routes) {
        if (c.food.length == 0 || c.constrictor) return 0.0;
        double best = 0.0;
        for (int f : c.food) {
            if (routes.cost[f] == INF) continue;   // inalcançável a partir daqui
            int cost = routes.cost[f], t = routes.turn[f];
            double value = 1.0 / (1 + cost);       // cost 0 = a comida está na casa do movimento
            int rt = c.arrT[f];
            if (rt >= 0 && (rt < t || (rt == t && c.arrLen[f] >= c.myLen))) value *= CONTESTED;
            if (c.isHazard[f] && c.urgency < 0.8) value *= HAZARD_FOOD;
            if (openNeighbors(c, f, t + 1) <= 1) value *= DEADEND_FOOD;
            best = Math.max(best, value);
        }
        boolean ahead = c.myLen > c.maxEnemyLen + 1;
        double base = ahead ? FOOD_BASE_AHEAD : FOOD_BASE;
        if (c.food.length <= SCARCE_FOOD) base = Math.min(1.0, base + 0.15);
        double mix = base + (1.0 - base) * c.urgency;
        return c.w.food * mix * best;
    }

    /** Penaliza vida baixa e o cenário 'não chego em nenhuma comida a tempo'. */
    static double evaluateHealth(Ctx c, int pos, Routes routes) {
        if (c.isFood[pos] && !c.constrictor) return 0.0;
        int h = healthAfterMove(c, pos);
        double penalty = 0.0;
        if (h < LOW_HEALTH) penalty += c.w.lowHealth * (LOW_HEALTH - h) / LOW_HEALTH;
        if (!c.constrictor && c.food.length > 0) {
            // custo em vida até a comida mais barata: se >= vida restante, morremos de fome antes
            int minCost = INF;
            boolean any = false;
            for (int f : c.food) {
                if (routes.cost[f] != INF) {
                    any = true;
                    minCost = Math.min(minCost, routes.cost[f]);
                }
            }
            if (any && minCost >= h) penalty += c.w.starvation;
            else if (!any && h <= 30) penalty += c.w.starvation / 2;
        }
        return -penalty;
    }

    /** Custo de entrar em hazard; maior quanto menos vida sobraria depois do dano. */
    static double evaluateHazards(Ctx c, int pos) {
        if (!c.isHazard[pos]) return 0.0;
        if (c.isFood[pos]) return -0.25 * c.w.hazard;   // comer reabastece a vida
        double risk = Math.min(1.0, Math.max(0.0, 1.0 - (double) healthAfterMove(c, pos) / MAX_HEALTH));
        return -c.w.hazard * (1.0 + 2.0 * risk);
    }

    /**
     * Para cada adversário que PODE entrar em 'pos' no próximo turno: maior -> perdemos (h2hLoss);
     * igual -> ambos morrem (h2hTie); menor -> bônus de kill, dividido pelo nº de opções dele
     * (se só tem 1 saída, é quase certo).
     */
    static double evaluateHeadToHead(Ctx c, int pos) {
        double score = 0.0;
        for (Enemy e : c.enemies) {
            boolean can = false;
            for (int n : e.nextCells) if (n == pos) can = true;
            if (!can) continue;
            if (e.length > c.myLen) score -= c.w.h2hLoss;
            else if (e.length == c.myLen) score -= c.w.h2hTie;
            else score += c.w.kill / Math.max(1, e.nextCells.length);
        }
        return score;
    }

    /** Pressão de proximidade: foge de cabeças maiores/iguais, persegue menores se saudável. */
    static double evaluateEnemies(Ctx c, int pos) {
        double score = 0.0;
        for (Enemy e : c.enemies) {
            int d = distance(c, pos, e.head);
            if (e.length >= c.myLen) {
                if (1 <= d && d <= THREAT_RANGE) score -= c.w.threat / d;
            } else if (d <= HUNT_RANGE && c.urgency < 0.5) {
                score += c.w.hunt / Math.max(1, d);
            }
        }
        return score;
    }

    /** Desempate posicional: centro bom, borda ruim (não se aplica a 'wrapped'). */
    static double evaluatePosition(Ctx c, int pos) {
        if (c.wrapped) return 0.0;
        double cx = (c.width - 1) / 2.0, cy = (c.height - 1) / 2.0;
        double maxD = cx + cy;
        int x = pos % c.width, y = pos / c.width;
        double center = maxD > 0 ? 1.0 - (Math.abs(x - cx) + Math.abs(y - cy)) / maxD : 1.0;
        double score = c.w.center * center * (1.0 - c.urgency);
        if (x == 0 || x == c.width - 1 || y == 0 || y == c.height - 1) score -= c.w.edge;
        return score;
    }

    static final String[] PART_NAMES = {"space", "territory", "mobility", "tail", "food", "health",
            "hazards", "h2h", "enemies", "position", "lookahead"};

    /** Nota clássica do movimento 'd' (índice de MOVE_ORDER). Preenche 'parts' se não for nulo. */
    static double evaluateMove(Ctx c, int d, double[] parts) {
        int pos = c.nbr[c.myHead][d];
        Flood flood = floodFill(c, pos, 1);
        Routes routes = dijkstraRoutes(c, pos, 1);
        double[] p = parts != null ? parts : new double[PART_NAMES.length];
        p[0] = evaluateSpace(c, flood);
        p[1] = evaluateTerritory(c, flood);
        p[2] = evaluateMobility(c, pos);
        p[3] = evaluateTail(c, flood);
        p[4] = evaluateFood(c, routes);
        p[5] = evaluateHealth(c, pos, routes);
        p[6] = evaluateHazards(c, pos);
        p[7] = evaluateHeadToHead(c, pos);
        p[8] = evaluateEnemies(c, pos);
        p[9] = evaluatePosition(c, pos);
        p[10] = 0.0;   // gancho para lookahead futuro (a busca já faz isso no 1v1)
        double sum = 0.0;
        for (double v : p) sum += v;
        return sum;
    }

    /** Avaliação barata, usada só se o orçamento de tempo estourar. */
    static double quickScore(Ctx c, int d) {
        int pos = c.nbr[c.myHead][d];
        return 10.0 * openNeighbors(c, pos, 2) + evaluateHeadToHead(c, pos);
    }

    // --------------------------------------------------------------------------- //
    // BUSCA (resumo; o algoritmo está na classe Search, no fim deste arquivo)
    // --------------------------------------------------------------------------- //

    /** Tempo de busca: 30% do timeout, no máximo SEARCH_CAP_MS (a jogada toda fica abaixo de 100 ms). */
    static double searchBudgetMs(GameState state, double timeoutMs) {
        if (searchMsOverride >= 0) return searchMsOverride;
        return Math.max(25.0, Math.min(SEARCH_CAP_MS, timeoutMs * SEARCH_TIME));
    }

    // --------------------------------------------------------------------------- //
    // FALLBACK — só roda se a lógica principal lançar exceção
    // --------------------------------------------------------------------------- //

    /** Paredes + todos os corpos (nossos e dos rivais). Respeita o modo 'wrapped'. */
    static String legacySafeMove(GameState state) {
        boolean wrapped = detectWrapped(state);
        int w = state.getBoard().getWidth(), h = state.getBoard().getHeight();
        java.util.Set<Long> occupied = new java.util.HashSet<>();
        for (Coordinate c : bodyOf(state.getYou())) occupied.add(c.getX() * 100000L + c.getY());
        if (state.getBoard().getSnakes() != null) {
            for (Snake s : state.getBoard().getSnakes()) {
                if (s == null) continue;
                for (Coordinate c : bodyOf(s)) occupied.add(c.getX() * 100000L + c.getY());
            }
        }
        Coordinate head = state.getYou().getHead() != null ? state.getYou().getHead() : bodyOf(state.getYou()).get(0);
        for (int d = 0; d < 4; d++) {
            int x = head.getX() + DX[d], y = head.getY() + DY[d];
            if (wrapped) {
                x = ((x % w) + w) % w;
                y = ((y % h) + h) % h;
            }
            if (x >= 0 && x < w && y >= 0 && y < h && !occupied.contains(x * 100000L + y)) return MOVE_ORDER[d];
        }
        return "up";
    }

    // --------------------------------------------------------------------------- //
    // PONTO DE ENTRADA
    // --------------------------------------------------------------------------- //

    /** Escolha pela pontuação clássica (todas as jogadas). */
    static int classicChoice(Ctx c, int[] possible) {
        double[] sc = new double[possible.length];
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < possible.length; i++) {
            sc[i] = System.nanoTime() > c.deadline ? quickScore(c, possible[i]) : evaluateMove(c, possible[i], null);
            if (sc[i] > bestScore) bestScore = sc[i];
        }
        List<Integer> best = new ArrayList<>();
        for (int i = 0; i < possible.length; i++) if (sc[i] >= bestScore - 1e-6) best.add(possible[i]);
        // ThreadLocalRandom (e não um Random estático): com SnapStart a semente de um campo estático
        // ficaria congelada no snapshot e toda instância restaurada sortearia a mesma sequência.
        return best.size() == 1 ? best.get(0) : best.get(ThreadLocalRandom.current().nextInt(best.size()));
    }

    /** Move-se com 'up', 'down', 'left' ou 'right'. Nunca lança exceção. */
    public static String getMove(GameState state) {
        long started = System.nanoTime();
        try {
            Ctx ctx = buildContext(state, started);
            int[] possible = getPossibleMoves(ctx);

            if (possible.length == 0) {
                String move = MOVE_ORDER[emergencyMove(ctx)];
                if (log) System.out.println("MOVE " + state.getTurn() + ": sem saída! emergência -> " + move);
                return move;
            }

            int chosen = -1;
            String how = "pontos";
            // 1v1 (uma rival) e mais de uma jogada legal: busca minimax. O prazo conta desde o início
            // deste getMove; a pontuação clássica só entra para desempatar (ou se a busca falhar).
            if (ctx.enemies.size() == 1 && possible.length >= 2) {
                try {
                    double timeoutMs = timeoutOf(state);
                    long deadline = started + (long) (searchBudgetMs(state, timeoutMs) * 1_000_000.0);
                    int[] order = sortByQuickScore(ctx, possible);
                    Search se = new Search(ctx, ctx.enemies.get(0));
                    se.search(se.makeState(ctx), order, deadline);
                    if (se.resCount > 0) {
                        double bestV = Double.NEGATIVE_INFINITY;
                        for (int k = 0; k < se.resCount; k++) bestV = Math.max(bestV, se.resVals[k]);
                        List<Integer> tied = new ArrayList<>();
                        for (int k = 0; k < se.resCount; k++) if (se.resVals[k] >= bestV - 1e-9) tied.add(se.resMoves[k]);
                        int pick = tied.get(0);
                        if (tied.size() > 1) {   // empate: a pontuação clássica decide (se ainda houver tempo)
                            long guard = started + (long) (Math.min(0.40 * timeoutMs, TIEBREAK_UNTIL_MS) * 1_000_000.0);
                            double bestKey = 0;
                            boolean first = true;
                            for (int m : tied) {
                                double k = System.nanoTime() < guard ? evaluateMove(ctx, m, null) : quickScore(ctx, m);
                                if (first || k > bestKey) {
                                    pick = m;
                                    bestKey = k;
                                    first = false;
                                }
                            }
                        }
                        chosen = pick;
                        how = "busca d=" + se.maxDepth + " nós=" + se.nodes + " v=" + Math.round(bestV);
                    }
                } catch (RuntimeException e) {
                    if (log) System.out.println("MOVE " + state.getTurn() + ": erro na busca, usando pontuação: " + e);
                }
            }

            if (chosen < 0) chosen = classicChoice(ctx, possible);

            if (log) {
                System.out.println("MOVE " + state.getTurn() + " [" + ctx.phase + "]: " + MOVE_ORDER[chosen] + " ("
                        + how + ", " + (System.nanoTime() - started) / 1_000_000 + " ms)");
            }
            return MOVE_ORDER[chosen];

        } catch (RuntimeException e) {   // nunca devolver erro: um movimento ruim é melhor que nenhum
            if (log) System.out.println("MOVE " + state.getTurn() + ": erro na lógica, usando fallback: " + e);
            try {
                return legacySafeMove(state);
            } catch (RuntimeException e2) {
                return "up";
            }
        }
    }

    /** Ordena as jogadas legais da mais para a menos promissora (ordenação estável). */
    static int[] sortByQuickScore(Ctx ctx, int[] possible) {
        int n = possible.length;
        double[] q = new double[n];
        for (int i = 0; i < n; i++) q[i] = quickScore(ctx, possible[i]);
        int[] out = possible.clone();
        for (int i = 1; i < n; i++) {   // inserção (estável): maiores primeiro
            int m = out[i];
            double v = q[i];
            int j = i - 1;
            while (j >= 0 && q[j] < v) {
                out[j + 1] = out[j];
                q[j + 1] = q[j];
                j--;
            }
            out[j + 1] = m;
            q[j + 1] = v;
        }
        return out;
    }

    // --------------------------------------------------------------------------- //
    // AQUECIMENTO — roda no início da Lambda (fase de inicialização) para a JVM compilar o código quente
    // --------------------------------------------------------------------------- //

    static void warmUp(long millis) {
        double saved = searchMsOverride;
        boolean savedLog = log;
        searchMsOverride = 12.0;
        log = false;
        try {
            long end = System.nanoTime() + millis * 1_000_000L;
            int round = 0;
            while (System.nanoTime() < end) {
                getMove(syntheticState(round++));
            }
        } finally {
            searchMsOverride = saved;
            log = savedLog;
        }
    }

    private static Coordinate pt(int x, int y) {
        Coordinate c = new Coordinate();
        c.setX(x);
        c.setY(y);
        return c;
    }

    private static Snake syntheticSnake(String id, int health, List<Coordinate> body) {
        Snake s = new Snake();
        s.setId(id);
        s.setHealth(health);
        s.setBody(body);
        s.setHead(body.get(0));
        s.setLength(body.size());
        return s;
    }

    /** Partida inventada de 2 cobras, só para a JVM compilar o código quente. */
    private static GameState syntheticState(int round) {
        int shift = round % 3;
        Snake me = syntheticSnake("a", 90 - 10 * shift, List.of(
                pt(5, 4 + shift), pt(4, 4 + shift), pt(3, 4 + shift), pt(3, 3 + shift)));
        Snake rival = syntheticSnake("b", 80, List.of(pt(7, 7), pt(8, 7), pt(9, 7)));

        Game game = new Game();
        game.setId("aquecimento");
        game.setTimeout(500);

        Board board = new Board();
        board.setWidth(11);
        board.setHeight(11);
        board.setFood(List.of(pt(6, 6), pt(1, 9), pt(5, 5)));
        board.setHazards(new ArrayList<>());
        board.setSnakes(List.of(me, rival));

        GameState st = new GameState();
        st.setGame(game);
        st.setBoard(board);
        st.setYou(me);
        st.setTurn(20 + round);
        return st;
    }

    // =============================================================================
    //  (A BUSCA FICA AQUI DENTRO DO Logic.java: o template pede toda a inteligência neste arquivo)
    // =============================================================================
    //  BUSCA: minimax (alfa-beta) com aprofundamento iterativo e avaliação por Voronoi
    // =============================================================================
    // Cada "lance" da árvore é um turno inteiro: eu escolho um movimento e a rival responde com o
    // pior para mim (visão pessimista); os dois são aplicados juntos com as regras reais (comida,
    // fome, hazard, parede, corpo, choque de cabeças).
    //
    // VELOCIDADE (v6): a avaliação usa bitboards (ver expand/expand2) e a árvore usa ordenação
    // "killer" (killerMine/killerTheirs). Os dois só deixam a conta mais rápida: a nota de cada
    // posição e o melhor lance são os mesmos da versão casa-por-casa; sobra tempo para olhar
    // mais lances à frente.
    //
    // Só roda no 1v1 (uma rival). Com 3+ cobras a busca com "demais cobras fixas" jogou PIOR que a
    // pontuação clássica nos testes (muitos choques de cabeça), então lá vale a pontuação.

    static final class Search {

        static final int WIN = 100000;
        static final double DRAW = 0.0;

        // Pesos da avaliação da busca (1 ponto = 1 casa de território de vantagem).
        // Por segmento a mais que a rival: ser maior ganha os choques de cabeça e o território.
        // v6: era 3. Nas partidas de teste (cobra contra ela mesma) 8 venceu 3, 15 venceu 8, 25 venceu 15,
        // e 40 empatou com 25 mas morreu mais no próprio corpo (arrisca demais). Ficou 25.
        static final double W_LENGTH = 25.0;
        static final double W_FOOD = 7.0;            // comida que chego antes do rival (decai com a distância)
        static final double W_FOOD_LOST = 2.5;       // comida que o rival chega antes
        static final double W_STARVE = 400.0;        // morro de fome antes de alcançar qualquer comida
        static final double W_STARVE_MARGIN = 6.0;   // por turno de folga abaixo de 8 até a comida mais próxima
        static final double W_CRAMPED = 6.0;         // por casa que falta para o território igualar meu tamanho
        static final double W_HAZARD = 20.0;         // cabeça dentro de hazard

        /** Estourou o prazo (sem stack trace: é só um sinal de controle). */
        static final class Timeout extends RuntimeException {
            private static final long serialVersionUID = 1L;

            Timeout() {
                super(null, null, false, false);
            }
        }

        private static final Timeout TIMEOUT = new Timeout();

        /** Estado do jogo dentro da busca (cobra 0 = eu, cobra 1 = rival). Imutável. */
        static final class St {
            final int[] b0, b1;     // corpos (cabeça no índice 0)
            final int h0, h1;       // vidas
            final int[] food;       // comidas

            St(int[] b0, int[] b1, int h0, int h1, int[] food) {
                this.b0 = b0;
                this.b1 = b1;
                this.h0 = h0;
                this.h1 = h1;
                this.food = food;
            }
        }

        final int W, H, V;
        final boolean wrapped, constrictor;
        final int hdmg;
        final int[][] nbr;
        final boolean[] hazard;
        final boolean anyHazard;
        final Logic.Enemy rival;

        long nodes;
        long deadline;
        int maxDepth;

        /** Resultado da última profundidade completa: movimentos (ordem testada) e valores. */
        int[] resMoves = new int[0];
        double[] resVals = new double[0];
        int resCount;

        // espaço de trabalho reaproveitado (a busca é de uma thread só)
        private final int[] cellt, seen;
        private final boolean[] foodFlag;
        private final int[] fr0, fr1, nx0, nx1;
        private int seenGen;

        // BITBOARDS: o tabuleiro como bits (bit c = casa c), em L palavras de 64 bits.
        // A avaliação (Voronoi) expande as fronteiras das duas cobras com deslocamentos de bits:
        // muito mais rápido que visitar casa por casa.
        private final int L;
        private final long lastMask;                        // bits válidos da última palavra
        private final long[] col0, colLast, row0, rowLast;  // máscaras das bordas
        private final long[] bFront0, bFront1, bNew0, bNew1, bClaimed, bFree, bFood, bT1, bT2;
        private final long[] bRel;                          // bRel[t]: casas que liberam no passo t

        // ORDENAÇÃO DE LANCES (não muda o resultado, só faz a poda alfa-beta cortar mais cedo):
        // o lance que foi melhor/refutou na mesma profundidade da árvore é testado primeiro.
        private final int[] killerMine = new int[128], killerTheirs = new int[128];

        // saída de step()
        private St stepSt;
        private boolean stepDead0, stepDead1;
        // saída de advance()
        private int advHealth;
        private boolean advAte;

        Search(Logic.Ctx ctx, Logic.Enemy rival) {
            this.W = ctx.width;
            this.H = ctx.height;
            this.V = ctx.V;
            this.wrapped = ctx.wrapped;
            this.constrictor = ctx.constrictor;
            this.hdmg = ctx.hazardDamage;
            this.nbr = ctx.nbr;
            this.hazard = ctx.isHazard;
            this.anyHazard = ctx.anyHazard;
            this.rival = rival;
            cellt = new int[V];
            seen = new int[V];
            foodFlag = new boolean[V];
            fr0 = new int[V];
            fr1 = new int[V];
            nx0 = new int[V];
            nx1 = new int[V];

            L = (V + 63) >>> 6;
            int rem = V & 63;
            lastMask = rem == 0 ? -1L : (1L << rem) - 1;
            col0 = new long[L];
            colLast = new long[L];
            row0 = new long[L];
            rowLast = new long[L];
            for (int c = 0; c < V; c++) {
                int x = c % W, y = c / W;
                if (x == 0) setBit(col0, c);
                if (x == W - 1) setBit(colLast, c);
                if (y == 0) setBit(row0, c);
                if (y == H - 1) setBit(rowLast, c);
            }
            bFront0 = new long[L];
            bFront1 = new long[L];
            bNew0 = new long[L];
            bNew1 = new long[L];
            bClaimed = new long[L];
            bFree = new long[L];
            bFood = new long[L];
            bT1 = new long[L];
            bT2 = new long[L];
            bRel = new long[(V + 3) * L];
            Arrays.fill(killerMine, -1);
            Arrays.fill(killerTheirs, -1);
        }

        // ------------------------------------------------------------- operações de bits

        private static void setBit(long[] a, int c) {
            a[c >>> 6] |= 1L << c;
        }

        /** dst = src deslocado k casas para cima no índice (c -> c + k). */
        private void shl(long[] src, int k, long[] dst) {
            int ws = k >>> 6, bs = k & 63;
            for (int i = L - 1; i >= 0; i--) {
                int j = i - ws;
                long v = j >= 0 ? src[j] << bs : 0L;
                if (bs != 0 && j >= 1) v |= src[j - 1] >>> (64 - bs);
                dst[i] = v;
            }
            dst[L - 1] &= lastMask;
        }

        /** dst = src deslocado k casas para baixo no índice (c -> c - k). */
        private void shr(long[] src, int k, long[] dst) {
            int ws = k >>> 6, bs = k & 63;
            for (int i = 0; i < L; i++) {
                int j = i + ws;
                long v = j < L ? src[j] >>> bs : 0L;
                if (bs != 0 && j + 1 < L) v |= src[j + 1] << (64 - bs);
                dst[i] = v;
            }
        }

        /** d = todas as casas vizinhas de s (cima, baixo, direita, esquerda; com a volta no 'wrapped'). */
        private void expand(long[] s, long[] d) {
            if (L == 2 && W < 64) {
                expand2(s, d);
                return;
            }
            if (L == 1 && W < 64) {
                long x = s[0];
                d[0] = ((x << W) | (x >>> W) | ((x & ~colLast[0]) << 1) | ((x & ~col0[0]) >>> 1)) & lastMask;
                if (wrapped) wrapPart(s, d);
                return;
            }
            long[] t1 = bT1, t2 = bT2;
            shl(s, W, d);                                         // cima: y + 1
            shr(s, W, t2);                                        // baixo: y - 1
            for (int i = 0; i < L; i++) d[i] |= t2[i];
            for (int i = 0; i < L; i++) t1[i] = s[i] & ~colLast[i];
            shl(t1, 1, t2);                                       // direita: x + 1
            for (int i = 0; i < L; i++) d[i] |= t2[i];
            for (int i = 0; i < L; i++) t1[i] = s[i] & ~col0[i];
            shr(t1, 1, t2);                                       // esquerda: x - 1
            for (int i = 0; i < L; i++) d[i] |= t2[i];
            if (wrapped) wrapPart(s, d);
        }

        /** Versão rápida para tabuleiros de até 128 casas (7x7, 11x11): duas palavras, sem laços. */
        private void expand2(long[] s, long[] d) {
            long lo = s[0], hi = s[1];
            int w = W, iw = 64 - W;
            long rlo = (lo << w) | (lo >>> w) | (hi << iw);                 // cima e baixo
            long rhi = (hi << w) | (lo >>> iw) | (hi >>> w);
            long a = lo & ~colLast[0], b = hi & ~colLast[1];                // direita
            rlo |= a << 1;
            rhi |= (b << 1) | (a >>> 63);
            a = lo & ~col0[0];                                              // esquerda
            b = hi & ~col0[1];
            rlo |= (a >>> 1) | (b << 63);
            rhi |= b >>> 1;
            d[0] = rlo;
            d[1] = rhi & lastMask;
            if (wrapped) wrapPart(s, d);
        }

        /** Vizinhos que "dão a volta" pelas bordas (só no modo wrapped). */
        private void wrapPart(long[] s, long[] d) {
            long[] t1 = bT1, t2 = bT2;
            {
                int jump = W * (H - 1);
                for (int i = 0; i < L; i++) t1[i] = s[i] & rowLast[i];
                shr(t1, jump, t2);                                // cima saindo do topo -> linha 0
                for (int i = 0; i < L; i++) d[i] |= t2[i];
                for (int i = 0; i < L; i++) t1[i] = s[i] & row0[i];
                shl(t1, jump, t2);                                // baixo saindo da base -> topo
                for (int i = 0; i < L; i++) d[i] |= t2[i];
                for (int i = 0; i < L; i++) t1[i] = s[i] & colLast[i];
                shr(t1, W - 1, t2);                               // direita saindo -> coluna 0
                for (int i = 0; i < L; i++) d[i] |= t2[i];
                for (int i = 0; i < L; i++) t1[i] = s[i] & col0[i];
                shl(t1, W - 1, t2);                               // esquerda saindo -> última coluna
                for (int i = 0; i < L; i++) d[i] |= t2[i];
            }
        }

        St makeState(Logic.Ctx ctx) {
            return new St(ctx.myBody.clone(), rival.body.clone(), ctx.myHealth, rival.health, ctx.food.clone());
        }

        // ------------------------------------------------------------------ regras

        private static boolean contains(int[] a, int from, int v) {
            for (int i = from; i < a.length; i++) if (a[i] == v) return true;
            return false;
        }

        /** Aplica um movimento a uma cobra. Devolve o corpo; vida e "comeu" saem em advHealth/advAte. */
        private int[] advance(int[] body, int health, int n, int[] food) {
            advAte = false;
            advHealth = health;
            if (n < 0) return body;
            if (constrictor) {
                int[] nb = new int[body.length + 1];
                nb[0] = n;
                System.arraycopy(body, 0, nb, 1, body.length);
                return nb;
            }
            health -= 1;
            if (hazard[n]) health -= hdmg;
            boolean ate = contains(food, 0, n);
            int len = body.length;
            // a engine primeiro move (a cauda sai) e depois, se comeu, duplica a NOVA cauda
            int[] moved = new int[ate ? len + 1 : len];
            moved[0] = n;
            System.arraycopy(body, 0, moved, 1, len - 1);
            if (ate) {
                moved[len] = moved[len - 1];
                advHealth = 100;
                advAte = true;
            } else {
                advHealth = health;
            }
            return moved;
        }

        /** Um turno com os dois movimentos. Resultado em stepSt / stepDead0 / stepDead1. */
        void step(St s, int m0, int m1) {
            int n0 = nbr[s.b0[0]][m0];
            int n1 = nbr[s.b1[0]][m1];
            int[] nb0 = advance(s.b0, s.h0, n0, s.food);
            int nh0 = advHealth;
            boolean e0 = advAte;
            int[] nb1 = advance(s.b1, s.h1, n1, s.food);
            int nh1 = advHealth;
            boolean e1 = advAte;
            boolean d0 = n0 < 0 || nh0 <= 0;
            boolean d1 = n1 < 0 || nh1 <= 0;
            if (!d0 && (contains(nb0, 1, n0) || contains(nb1, 1, n0))) d0 = true;
            if (!d1 && (contains(nb1, 1, n1) || contains(nb0, 1, n1))) d1 = true;
            if (n0 == n1 && n0 >= 0) {   // choque de cabeças: o menor morre, igual morrem os dois
                int l0 = nb0.length, l1 = nb1.length;
                if (l0 <= l1) d0 = true;
                if (l1 <= l0) d1 = true;
            }
            int[] food = s.food;
            if (e0 || e1) {
                int[] tmp = new int[food.length];
                int k = 0;
                for (int f : food) {
                    if ((e0 && f == n0) || (e1 && f == n1)) continue;
                    tmp[k++] = f;
                }
                food = Arrays.copyOf(tmp, k);
            }
            stepSt = new St(nb0, nb1, nh0, nh1, food);
            stepDead0 = d0;
            stepDead1 = d1;
        }

        /** Máscara dos movimentos que não matam de imediato (parede/corpo; caudas que saem contam como livres). */
        int moves(int[] body, int[] other) {
            int head = body[0];
            int mask = 0;
            for (int m = 0; m < 4; m++) {
                int n = nbr[head][m];
                if (n < 0) continue;
                boolean occ = false;
                for (int i = 0; i < body.length - 1 && !occ; i++) if (body[i] == n) occ = true;
                for (int i = 0; i < other.length - 1 && !occ; i++) if (other[i] == n) occ = true;
                if (!occ) mask |= 1 << m;
            }
            return mask == 0 ? 1 : mask;   // sem saída: devolve um movimento qualquer (a simulação o marca como morte)
        }

        // ------------------------------------------------------------- avaliação

        /** Casa ocupada que ainda não foi liberada no passo t (constrictor: nunca libera). */
        private boolean blocked(int c, int t) {
            int ct = cellt[c];
            return ct > 0 && (constrictor || ct > t);
        }

        /** Menor distância (em turnos) até alguma comida, ignorando os rivais. -1 se não houver. */
        private int foodDist(int head, int limit) {
            int sg = ++seenGen;
            seen[head] = sg;
            int fc = 1;
            int[] front = fr0, next = nx0;
            front[0] = head;
            int t = 0;
            while (fc > 0 && t < limit) {
                t++;
                int nc = 0;
                boolean found = false;
                for (int i = 0; i < fc; i++) {
                    for (int n : nbr[front[i]]) {
                        if (n < 0 || seen[n] == sg || blocked(n, t)) continue;
                        seen[n] = sg;
                        next[nc++] = n;
                        if (foodFlag[n]) found = true;
                    }
                }
                if (found) return t;
                int[] tmp = front;
                front = next;
                next = tmp;
                fc = nc;
            }
            return -1;
        }

        /**
         * Custo em PONTOS DE VIDA até a comida mais barata (hazard custa 1 + dano por casa). Só é usado
         * quando há hazards e a vida está baixa. -1 se não alcança com a vida que tem.
         */
        private int foodLifeCost(int head, int hp) {
            int[] best = new int[V];
            Arrays.fill(best, 1_000_000_000);
            best[head] = 0;
            PriorityQueue<Long> heap = new PriorityQueue<>();
            heap.add(pack(0, 0, head));
            while (!heap.isEmpty()) {
                long k = heap.poll();
                int cost = (int) (k >>> 40), steps = (int) ((k >>> 20) & 0xFFFFF), c = (int) (k & 0xFFFFF);
                if (cost > best[c]) continue;
                if (foodFlag[c] && c != head) return cost - (hazard[c] ? hdmg : 0);
                if (cost >= hp) continue;
                for (int n : nbr[c]) {
                    if (n < 0 || cellt[n] > steps + 1) continue;
                    int nc = cost + 1 + (hazard[n] ? hdmg : 0);
                    if (nc < best[n]) {
                        best[n] = nc;
                        heap.add(pack(nc, steps + 1, n));
                    }
                }
            }
            return -1;
        }

        private static long pack(int cost, int steps, int cell) {
            return ((long) cost << 40) | ((long) steps << 20) | cell;
        }

        /** Nota do estado do ponto de vista da cobra 0 (eu): positivo = bom para mim. */
        double evaluate(St s) {
            int[] b0 = s.b0, b1 = s.b1;
            int l0 = b0.length, l1 = b1.length;
            // em que turno cada casa ocupada fica livre (tamanho - posição; duplicata de cauda = maior valor)
            for (int i = 0; i < l0; i++) {
                int t = l0 - i;
                if (t > cellt[b0[i]]) cellt[b0[i]] = t;
            }
            for (int i = 0; i < l1; i++) {
                int t = l1 - i;
                if (t > cellt[b1[i]]) cellt[b1[i]] = t;
            }
            boolean anyFood = s.food.length > 0;
            for (int f : s.food) foodFlag[f] = true;

            int h0c = b0[0], h1c = b1[0];
            // casas livres agora; bRel[t] = casas que ficam livres no passo t (caudas saindo)
            long[] free = bFree, food = bFood, claimed = bClaimed;
            Arrays.fill(free, 0, L, -1L);
            free[L - 1] = lastMask;
            Arrays.fill(food, 0L);
            for (int f : s.food) setBit(food, f);
            int maxT = Math.max(l0, l1) + 1;
            if (!constrictor) Arrays.fill(bRel, 0, (maxT + 1) * L, 0L);
            for (int pass = 0; pass < 2; pass++) {
                int[] body = pass == 0 ? b0 : b1;
                for (int c : body) {
                    free[c >>> 6] &= ~(1L << c);
                    if (!constrictor) bRel[cellt[c] * L + (c >>> 6)] |= 1L << c;
                }
            }
            long[] f0 = bFront0, f1 = bFront1, n0 = bNew0, n1 = bNew1;
            Arrays.fill(f0, 0L);
            Arrays.fill(f1, 0L);
            Arrays.fill(claimed, 0L);
            setBit(f0, h0c);
            setBit(f1, h1c);
            setBit(claimed, h0c);
            setBit(claimed, h1c);
            boolean live0 = true, live1 = true;
            int c0 = 0, c1 = 0;
            double foodSc = 0.0;
            int t = 0;
            // Voronoi: a cada passo cada cobra expande a fronteira; casa alcançada por uma só é dela;
            // disputada (mesmo turno) fica com a MAIOR (a menor perderia o choque); igual = de ninguém.
            while (live0 || live1) {
                t++;
                if (!constrictor && t <= maxT) {
                    int base = t * L;
                    for (int i = 0; i < L; i++) free[i] |= bRel[base + i];
                }
                if (live0) expand(f0, n0); else Arrays.fill(n0, 0L);
                if (live1) expand(f1, n1); else Arrays.fill(n1, 0L);
                int a = 0, b = 0;
                boolean any0 = false, any1 = false;
                for (int i = 0; i < L; i++) {
                    long open = free[i] & ~claimed[i];
                    long x0 = n0[i] & open, x1 = n1[i] & open;
                    n0[i] = x0;
                    n1[i] = x1;
                    long both = x0 & x1;
                    long o0 = l0 > l1 ? x0 : x0 & ~both;
                    long o1 = l1 > l0 ? x1 : x1 & ~both;
                    c0 += Long.bitCount(o0);
                    c1 += Long.bitCount(o1);
                    a += Long.bitCount(o0 & food[i]);
                    b += Long.bitCount(o1 & food[i]);
                    claimed[i] |= x0 | x1;
                    if (x0 != 0) any0 = true;
                    if (x1 != 0) any1 = true;
                }
                if (anyFood) {
                    if (a > 0) foodSc += W_FOOD * a / (1 + t);
                    if (b > 0) foodSc -= W_FOOD_LOST * b / (1 + t);
                }
                long[] tmp = f0;
                f0 = n0;
                n0 = tmp;
                tmp = f1;
                f1 = n1;
                n1 = tmp;
                live0 = any0;
                live1 = any1;
            }

            double sc = (c0 - c1) + W_LENGTH * (l0 - l1) + foodSc;
            if (c0 < l0) sc -= W_CRAMPED * (l0 - c0);
            if (c1 < l1) sc += W_CRAMPED * (l1 - c1);

            if (!constrictor) {
                for (int who = 0; who < 2; who++) {
                    int hp = who == 0 ? s.h0 : s.h1;
                    if (hp >= 45) continue;
                    int head = who == 0 ? h0c : h1c;
                    int d;
                    if (anyFood && anyHazard) d = foodLifeCost(head, hp);
                    else d = anyFood ? foodDist(head, hp) : -1;
                    double pen;
                    if (d < 0) {
                        // há comida mas não alcanço (ou não há comida e a vida está no fim)
                        pen = anyFood ? W_STARVE * (45 - hp) / 45 : (hp <= 5 ? W_STARVE : 0.0);
                    } else {
                        int margin = hp - d;
                        pen = margin <= 0 ? W_STARVE : W_STARVE_MARGIN * Math.max(0, 8 - margin);
                    }
                    sc += who == 0 ? -pen : pen;
                }
            }
            if (anyHazard) {
                if (hazard[h0c]) sc -= W_HAZARD;
                if (hazard[h1c]) sc += W_HAZARD;
            }

            // limpa o espaço de trabalho
            for (int c : b0) cellt[c] = 0;
            for (int c : b1) cellt[c] = 0;
            for (int f : s.food) foodFlag[f] = false;
            return sc;
        }

        // ----------------------------------------------------------------- busca

        private void tick() {
            nodes++;
            if ((nodes & 31) == 0 && System.nanoTime() > deadline) throw TIMEOUT;
        }

        private double child(St s, int m0, int m1, int depth, double alpha, double beta, int ply) {
            step(s, m0, m1);
            St ns = stepSt;
            if (stepDead0) return stepDead1 ? DRAW : -(WIN - ply);   // os dois morrem = empate
            if (stepDead1) return WIN - ply;                         // quanto mais cedo a vitória, melhor
            return ab(ns, depth - 1, alpha, beta, ply + 1);
        }

        private double ab(St s, int depth, double alpha, double beta, int ply) {
            tick();
            if (depth <= 0) return evaluate(s);
            int mine = moves(s.b0, s.b1);
            int theirs = moves(s.b1, s.b0);
            int km = killerMine[ply], kt = killerTheirs[ply];
            double best = -10.0 * WIN;
            int bestM0 = -1;
            // testa primeiro o lance "killer" (o melhor da última vez nesta profundidade), depois os outros
            for (int i0 = 0; i0 < 5; i0++) {
                int m0 = i0 == 0 ? km : i0 - 1;
                if (m0 < 0 || (i0 > 0 && m0 == km) || (mine & (1 << m0)) == 0) continue;
                double worst = 10.0 * WIN;
                int worstM1 = -1;
                for (int i1 = 0; i1 < 5; i1++) {
                    int m1 = i1 == 0 ? kt : i1 - 1;
                    if (m1 < 0 || (i1 > 0 && m1 == kt) || (theirs & (1 << m1)) == 0) continue;
                    double v = child(s, m0, m1, depth, alpha, Math.min(beta, worst), ply);
                    if (v < worst) {
                        worst = v;
                        worstM1 = m1;
                    }
                    if (worst <= alpha) break;
                }
                if (worstM1 >= 0) killerTheirs[ply] = worstM1;   // a resposta que mais me prejudicou
                if (worst > best) {
                    best = worst;
                    bestM0 = m0;
                }
                if (best > alpha) alpha = best;
                if (alpha >= beta) break;
            }
            if (bestM0 >= 0) killerMine[ply] = bestM0;
            return best;
        }

        /** Valores da raiz numa profundidade fixa, sem limite de tempo (usado pelos testes de paridade). */
        double[] rootValues(St s, int[] order, int depth) {
            deadline = Long.MAX_VALUE;
            int theirs = moves(s.b1, s.b0);
            double[] vals = new double[order.length];
            double alpha = -10.0 * WIN;
            for (int k = 0; k < order.length; k++) {
                double worst = 10.0 * WIN;
                for (int m1 = 0; m1 < 4; m1++) {
                    if ((theirs & (1 << m1)) == 0) continue;
                    double v = child(s, order[k], m1, depth, alpha, worst, 0);
                    if (v < worst) worst = v;
                    if (worst <= alpha) break;
                }
                vals[k] = worst;
                if (worst > alpha) alpha = worst;
            }
            return vals;
        }

        /** Aprofundamento iterativo. O resultado da última profundidade completa fica em resMoves/resVals. */
        void search(St s, int[] order, long deadlineNanos) {
            this.deadline = deadlineNanos;
            resCount = 0;
            int theirs = moves(s.b1, s.b0);
            for (int depth = 1; depth < 60; depth++) {
                int n = order.length;
                int[] mv = new int[n];
                double[] val = new double[n];
                double alpha = -10.0 * WIN;
                try {
                    for (int k = 0; k < n; k++) {
                        int m0 = order[k];
                        double worst = 10.0 * WIN;
                        for (int m1 = 0; m1 < 4; m1++) {
                            if ((theirs & (1 << m1)) == 0) continue;
                            double v = child(s, m0, m1, depth, alpha, worst, 0);
                            if (v < worst) worst = v;
                            if (worst <= alpha) break;
                        }
                        mv[k] = m0;
                        val[k] = worst;
                        if (worst > alpha) alpha = worst;
                    }
                } catch (Timeout e) {
                    break;
                }
                resMoves = mv;
                resVals = val;
                resCount = n;
                maxDepth = depth;
                // próxima profundidade: melhores primeiro (ordenação estável por valor decrescente)
                int[] nextOrder = mv.clone();
                double[] nv = val.clone();
                for (int i = 1; i < n; i++) {
                    int m = nextOrder[i];
                    double v = nv[i];
                    int j = i - 1;
                    while (j >= 0 && nv[j] < v) {
                        nextOrder[j + 1] = nextOrder[j];
                        nv[j + 1] = nv[j];
                        j--;
                    }
                    nextOrder[j + 1] = m;
                    nv[j + 1] = v;
                }
                order = nextOrder;
                double mx = Double.NEGATIVE_INFINITY;
                for (double v : val) mx = Math.max(mx, v);
                if (mx >= WIN - 200 || mx <= -(WIN - 200)) break;   // vitória ou derrota forçada: aprofundar não muda nada
            }
        }
    }
}
