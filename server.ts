import dotenv from 'dotenv';
dotenv.config({ override: true });
import express from 'express';
import http from 'http';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { spawn, execFileSync } from 'child_process';
import readline from 'readline';
import { createServer as createViteServer } from 'vite';
import type { Duplex } from 'stream';

const PORT = 3000;
const WS_MAGIC_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

/**
 * Encode a UTF-8 JSON string as an RFC 6455 WebSocket text frame (server -> client, unmasked).
 */
function encodeWsTextFrame(text: string): Buffer {
  const payload = Buffer.from(text, 'utf-8');
  const len = payload.length;
  let header: Buffer;
  if (len < 126) {
    header = Buffer.alloc(2);
    header[0] = 0x81; // FIN + text opcode
    header[1] = len;
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

function encodeWsPongFrame(payload: Buffer): Buffer {
  const len = Math.min(payload.length, 125);
  const frame = Buffer.alloc(2 + len);
  frame[0] = 0x8a; // FIN + pong opcode
  frame[1] = len;
  if (len > 0) {
    payload.copy(frame, 2, 0, len);
  }
  return frame;
}

/**
 * Decode incoming RFC 6455 masked client frames from a raw buffer.
 */
function decodeWsFrames(buffer: Buffer): {
  messages: string[];
  pongs: Buffer[];
  remaining: Buffer;
  closed: boolean;
} {
  const messages: string[] = [];
  const pongs: Buffer[] = [];
  let offset = 0;
  let closed = false;

  while (offset + 2 <= buffer.length) {
    const byte0 = buffer[offset];
    const byte1 = buffer[offset + 1];
    const opcode = byte0 & 0x0f;
    const isMasked = (byte1 & 0x80) !== 0;
    let payloadLen = byte1 & 0x7f;
    let headerLen = 2;

    if (payloadLen === 126) {
      if (offset + 4 > buffer.length) break;
      payloadLen = buffer.readUInt16BE(offset + 2);
      headerLen = 4;
    } else if (payloadLen === 127) {
      if (offset + 10 > buffer.length) break;
      payloadLen = Number(buffer.readBigUInt64BE(offset + 2));
      headerLen = 10;
    }

    const maskLen = isMasked ? 4 : 0;
    const totalFrameLen = headerLen + maskLen + payloadLen;
    if (offset + totalFrameLen > buffer.length) {
      break;
    }

    if (opcode === 0x08) {
      closed = true;
      offset += totalFrameLen;
      break;
    }

    const payloadStart = offset + headerLen + maskLen;
    const payload = Buffer.from(buffer.subarray(payloadStart, payloadStart + payloadLen));
    if (isMasked) {
      const mask = buffer.subarray(offset + headerLen, offset + headerLen + 4);
      for (let i = 0; i < payload.length; i++) {
        payload[i] ^= mask[i % 4];
      }
    }

    if (opcode === 0x09) {
      pongs.push(encodeWsPongFrame(payload));
    } else if (opcode === 0x01) {
      messages.push(payload.toString('utf-8'));
    }
    offset += totalFrameLen;
  }

  return {
    messages,
    pongs,
    remaining: buffer.subarray(offset),
    closed,
  };
}

async function startServer() {
  const pythonBin = process.platform === 'win32' ? (process.env.PYTHON || 'python') : 'python3';

  // Ensure pre-synthesized 16 kHz mono PCM WAV placeholder files and manifest.json exist on disk
  try {
    execFileSync(pythonBin, ['-m', 'backend.app.scripts.presynthesize_instructions'], {
      cwd: process.cwd(),
      stdio: 'inherit',
    });
  } catch (err) {
    console.error('Warning: presynthesize_instructions.py failed:', err);
  }

  // Spawn persistent Python backend bridge (backend/app/main.py --stdio-ws-bridge)
  const pyProc = spawn(pythonBin, ['-m', 'backend.app.main', '--stdio-ws-bridge'], {
    cwd: process.cwd(),
    stdio: ['pipe', 'pipe', 'inherit'],
  });

  const pendingCallbacks = new Map<string, (responseJson: string) => void>();
  const streamCallbacks = new Map<string, (chunkJson: string) => void>();
  let reqCounter = 0;

  if (pyProc.stdout) {
    const rl = readline.createInterface({ input: pyProc.stdout });
    rl.on('line', (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const parsed = JSON.parse(trimmed);
        const reqId = parsed.req_id;
        if (parsed.stream_chunk === true) {
          if (reqId && streamCallbacks.has(reqId)) {
            streamCallbacks.get(reqId)!(trimmed);
          }
          return;
        }
        if (reqId && pendingCallbacks.has(reqId)) {
          streamCallbacks.delete(reqId);
          const cb = pendingCallbacks.get(reqId)!;
          pendingCallbacks.delete(reqId);
          cb(trimmed);
        }
      } catch {
        // Ignore non-JSON stdout
      }
    });
  }

  function sendToPythonBridge(
    messageObj: Record<string, unknown>,
    onStreamChunk?: (chunkJson: string) => void,
  ): Promise<string> {
    return new Promise((resolve) => {
      const reqId = `ws-req-${++reqCounter}-${Date.now()}`;
      const enriched = { ...messageObj, req_id: reqId };
      const timeoutHandle = setTimeout(() => {
        if (pendingCallbacks.has(reqId)) {
          pendingCallbacks.delete(reqId);
          streamCallbacks.delete(reqId);
          resolve(
            JSON.stringify({
              type: 'error',
              req_id: reqId,
              reason_code: 'bridge_request_timeout',
              message: 'Python session bridge timed out processing request.',
            }),
          );
        }
      }, 16000);
      pendingCallbacks.set(reqId, (responseJson: string) => {
        clearTimeout(timeoutHandle);
        resolve(responseJson);
      });
      if (onStreamChunk) {
        streamCallbacks.set(reqId, onStreamChunk);
      }
      pyProc.stdin?.write(JSON.stringify(enriched) + '\n');
    });
  }

  const app = express();

  // Serve synthetic demo fleet live status from backend Python bridge
  app.get('/api/demo-fleet', async (_req, res) => {
    try {
      const replyRaw = await sendToPythonBridge({ type: 'get_demo_fleet' });
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.send(replyRaw);
    } catch (err) {
      res.status(500).json({ error: 'demo_fleet_unavailable', message: String(err) });
    }
  });

  // Serve authoritative structured incident handoff for Handoff View & PDF export
  app.get('/api/session/:sessionId/handoff', async (req, res) => {
    try {
      const replyRaw = await sendToPythonBridge({
        type: 'get_handoff',
        session_id: req.params.sessionId,
      });
      const parsed = JSON.parse(replyRaw);
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      if (parsed.type === 'error') {
        res.status(404).send(replyRaw);
        return;
      }
      res.send(replyRaw);
    } catch (err) {
      res.status(500).json({ error: 'handoff_unavailable', message: String(err) });
    }
  });

  // Serve pre-synthesized instruction manifest (Section 15.2)
  app.get('/api/instructions/manifest', (_req, res) => {
    const manifestPath = path.resolve(process.cwd(), 'backend-media/instructions/manifest.json');
    if (fs.existsSync(manifestPath)) {
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.sendFile(manifestPath);
      return;
    }
    res.status(404).json({ error: 'manifest_not_found' });
  });

  // Serve pre-synthesized 16 kHz mono PCM WAV files from backend-media/instructions/
  app.use(
    '/backend-media/instructions',
    express.static(path.resolve(process.cwd(), 'backend-media/instructions'), {
      setHeaders: (res) => {
        res.setHeader('Content-Type', 'audio/wav');
        res.setHeader('Cache-Control', 'no-cache');
      },
    }),
  );

  // Ensure ONNX Runtime WebAssembly binaries (.wasm) and glue (.mjs) exist in public/onnx-wasm
  const publicOnnxDir = path.resolve(process.cwd(), 'public/onnx-wasm');
  const nodeModulesOnnxDir = path.resolve(process.cwd(), 'node_modules/onnxruntime-web/dist');
  try {
    if (fs.existsSync(nodeModulesOnnxDir)) {
      fs.mkdirSync(publicOnnxDir, { recursive: true });
      for (const entry of fs.readdirSync(nodeModulesOnnxDir)) {
        if (entry.startsWith('ort-wasm') && (entry.endsWith('.wasm') || entry.endsWith('.mjs'))) {
          const srcFile = path.join(nodeModulesOnnxDir, entry);
          const destFile = path.join(publicOnnxDir, entry);
          if (!fs.existsSync(destFile)) {
            fs.copyFileSync(srcFile, destFile);
          }
        }
      }
    }
  } catch (err) {
    console.warn('Notice: failed to sync onnx-wasm assets:', err);
  }

  const setOnnxHeaders = (res: express.Response, filePath: string) => {
    if (filePath.endsWith('.mjs') || filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'text/javascript; charset=utf-8');
    } else if (filePath.endsWith('.wasm')) {
      res.setHeader('Content-Type', 'application/wasm');
    }
    res.setHeader('Cache-Control', 'no-cache');
  };

  // Serve ONNX Runtime WebAssembly binaries and .mjs glue directly so Vite dev middleware does not 404 on `?import` or fall through to index.html
  app.use('/onnx-wasm', express.static(publicOnnxDir, { setHeaders: setOnnxHeaders }));
  app.use('/onnx-wasm', express.static(nodeModulesOnnxDir, { setHeaders: setOnnxHeaders }));
  app.use('/onnx-wasm', (_req, res) => {
    res.status(404).end();
  });

  const server = http.createServer(app);

  // Attach Vite middleware in development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*all', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  // Handle WebSocket /ws/session upgrades and route frames to Python backend/app/main.py
  server.on('upgrade', (req: http.IncomingMessage, socket: Duplex, head: Buffer) => {
    const reqUrl = req.url || '';
    if (!reqUrl.startsWith('/ws/session')) {
      socket.destroy();
      return;
    }

    const key = req.headers['sec-websocket-key'];
    if (!key || typeof key !== 'string') {
      socket.destroy();
      return;
    }

    const acceptKey = crypto
      .createHash('sha1')
      .update(key + WS_MAGIC_GUID)
      .digest('base64');

    const responseHeaders = [
      'HTTP/1.1 101 Switching Protocols',
      'Upgrade: websocket',
      'Connection: Upgrade',
      `Sec-WebSocket-Accept: ${acceptKey}`,
    ];
    socket.write(responseHeaders.join('\r\n') + '\r\n\r\n');

    let recvBuffer: Buffer = head && head.length > 0 ? Buffer.from(head) : Buffer.alloc(0);
    let socketQueue: Promise<void> = Promise.resolve();

    socket.on('data', (chunk: Buffer) => {
      recvBuffer = Buffer.concat([recvBuffer, chunk]);
      const decoded = decodeWsFrames(recvBuffer);
      recvBuffer = decoded.remaining;

      for (const pongFrame of decoded.pongs) {
        if (!socket.destroyed) {
          socket.write(pongFrame);
        }
      }

      if (decoded.closed) {
        socket.end();
        return;
      }

      if (decoded.messages.length === 0) {
        return;
      }

      const dispatchWsPayload = async (rawMsg: string) => {
        if (socket.destroyed) return;
        try {
          const parsed = JSON.parse(rawMsg);
          const pyReplyJson = await sendToPythonBridge(parsed, (chunkJson) => {
            if (!socket.destroyed) {
              socket.write(encodeWsTextFrame(chunkJson));
            }
          });
          if (!socket.destroyed) {
            socket.write(encodeWsTextFrame(pyReplyJson));
          }
        } catch (err) {
          if (!socket.destroyed) {
            socket.write(
              encodeWsTextFrame(
                JSON.stringify({
                  type: 'error',
                  reason_code: 'ws_gateway_error',
                  message: String(err),
                }),
              ),
            );
          }
        }
      };

      const orderedBatch: string[] = [];
      for (const rawMsg of decoded.messages) {
        let msgType = '';
        try {
          msgType = String((JSON.parse(rawMsg) as { type?: unknown }).type ?? '');
        } catch {
          msgType = '';
        }
        // Background lifecycle/control events bypass PCM chunk queue
        if (
          msgType === 'location_confirm' ||
          msgType === 'demo_lifecycle_tick' ||
          msgType === 'session_end'
        ) {
          void dispatchWsPayload(rawMsg);
        } else {
          orderedBatch.push(rawMsg);
        }
      }

      if (orderedBatch.length > 0) {
        socketQueue = socketQueue.then(async () => {
          for (const rawMsg of orderedBatch) {
            await dispatchWsPayload(rawMsg);
          }
        });
      }
    });

    socket.on('error', () => {
      socket.destroy();
    });
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`SETU Gateway + Python /ws/session Bridge listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
