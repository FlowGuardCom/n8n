import express from 'express';
import bodyParser from 'body-parser';
import axios from 'axios';

const app = express();
app.use(bodyParser.json());

app.get('/healthz', (_req, res) => res.json({status: 'ok'}));

app.post('/log-eval', async (req, res) => {
  try {
    const payload = req.body;

    // TODO: aquí podrás:
    //  - llamar a la REST API de W&B para crear/run/log (con tu WANDB_API_KEY)
    //  - o delegar a un worker Python para usar Weave/EvaluationLogger

    console.log('[weave-logger] payload recibido:', JSON.stringify(payload));
    res.json({ status: 'ok', received: payload });
  } catch (err) {
    console.error('[weave-logger] error:', err);
    res.status(500).json({ error: err?.message || String(err) });
  }
});

app.listen(3000, '0.0.0.0', () => {
  console.log('weave-logger escuchando en puerto 3000');
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`weave-logger escuchando en puerto ${PORT}`);
});