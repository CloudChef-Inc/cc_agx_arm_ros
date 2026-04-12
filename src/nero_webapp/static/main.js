// Dual-Nero browser control.
//
// Renders two actual Nero arms (not stick figures) using urdf-loader + the
// DAE visual meshes shipped by agx_arm_description. Commands and feedback
// flow over a WebSocket at /ws; sliders are built from /joint_limits.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import URDFLoader from "urdf-loader";

// ------------ scene convention ------------------------------------------
// World axes (overrides three.js default Y-up):
//   +X = robot's RIGHT          -X = robot's LEFT
//   +Y = robot's FRONT          -Y = robot's BACK
//   +Z = UP                     -Z = DOWN
//
// Right arm mounts at +X and extends along +X.
// Left arm  mounts at -X and extends along -X.
const TORSO_X = 0.30;  // width  (left↔right)
const TORSO_Y = 0.10;  // depth  (front↔back)
const TORSO_Z = 0.60;  // height (down↔up)
const SHOULDER_X = 0.15; // lateral offset of shoulder from centerline
const SHOULDER_Z = 0.55; // shoulder height above floor

// ------------ DOM bootstrap ---------------------------------------------
const statusEl = document.getElementById("status");
const viewport = document.getElementById("viewport");
const panels = {
  left:  document.querySelector('.panel[data-side="left"]'),
  right: document.querySelector('.panel[data-side="right"]'),
};

let jointNames = [];
let jointLimits = [];
const sliderState = {
  left:  { cmd: [], fb: [], rowEls: [] },
  right: { cmd: [], fb: [], rowEls: [] },
};
const armRobots = { left: null, right: null };

// ------------ three.js scene --------------------------------------------
// Switch the world to Z-up before creating any object whose orientation
// depends on the up vector (camera, OrbitControls).
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const scene = new THREE.Scene();
// Medium blue-grey: contrasts with the silver/black Nero meshes.
scene.background = new THREE.Color(0x3a4855);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
camera.up.set(0, 0, 1);
// Position the camera in front-right of the robot, slightly above shoulder.
camera.position.set(1.8, -2.2, 1.4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, SHOULDER_Z);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(2, -2, 3);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.35);
fill.position.set(-2, 2, 1);
scene.add(fill);

// Floor grid: GridHelper is in the XZ plane by default (for Y-up worlds).
// Rotate it 90° around X so it lies in the XY plane (the floor in Z-up).
const grid = new THREE.GridHelper(4, 16, 0x6b7785, 0x4a5460);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

// Torso pillar standing along +Z. BoxGeometry params are (X, Y, Z).
const torso = new THREE.Mesh(
  new THREE.BoxGeometry(TORSO_X, TORSO_Y, TORSO_Z),
  new THREE.MeshStandardMaterial({ color: 0x1f242b, roughness: 0.7 }),
);
torso.position.set(0, 0, TORSO_Z / 2);
scene.add(torso);

// ------------ axes + front marker ---------------------------------------
// Build a labeled arrow for each of ±X, ±Y, ±Z so the user can describe
// orientation problems unambiguously.
function makeTextSprite(text, color = "#ffffff") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(0,0,0,0)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = "bold 64px -apple-system, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 6;
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.strokeText(text, 128, 64);
  ctx.fillStyle = color;
  ctx.fillText(text, 128, 64);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.20, 0.10, 1);
  sprite.renderOrder = 999;
  return sprite;
}

function addAxisArrow(dir, length, color, label) {
  const v = new THREE.Vector3(...dir).normalize();
  const arrow = new THREE.ArrowHelper(
    v,
    new THREE.Vector3(0, 0, 0),
    length,
    color,
    length * 0.18,
    length * 0.10,
  );
  scene.add(arrow);
  const tip = v.clone().multiplyScalar(length * 1.18);
  const sprite = makeTextSprite(label, "#" + color.toString(16).padStart(6, "0"));
  sprite.position.copy(tip);
  scene.add(sprite);
}

const AXIS_LEN = 0.45;
addAxisArrow([ 1, 0, 0], AXIS_LEN,        0xff5a5a, "+X (right)");
addAxisArrow([-1, 0, 0], AXIS_LEN * 0.7,  0x884040, "-X (left)");
addAxisArrow([ 0, 1, 0], AXIS_LEN,        0x5aff7a, "+Y (front)");
addAxisArrow([ 0,-1, 0], AXIS_LEN * 0.7,  0x408840, "-Y (back)");
addAxisArrow([ 0, 0, 1], AXIS_LEN,        0x5aaaff, "+Z (up)");
addAxisArrow([ 0, 0,-1], AXIS_LEN * 0.7,  0x405588, "-Z (down)");

// "FRONT" tag floating in front of the torso along +Y so it's obvious
// which direction the robot is supposed to face.
const frontTag = makeTextSprite("FRONT (+Y)", "#ffd56a");
frontTag.position.set(0, 0.75, 0.05);
frontTag.scale.set(0.50, 0.25, 1);
scene.add(frontTag);

// ------------ URDF loading ----------------------------------------------
const loadingManager = new THREE.LoadingManager();
const urdfLoader = new URDFLoader(loadingManager);
urdfLoader.packages = {
  agx_arm_description: "/pkg/agx_arm_description",
};

// Custom mesh loader: urdf-loader's defaults may not resolve three/addons
// loaders through our importmap, so we handle DAE + STL ourselves.
urdfLoader.loadMeshCb = (path, manager, done) => {
  const ext = path.split(".").pop().toLowerCase();
  if (ext === "dae") {
    new ColladaLoader(manager).load(
      path,
      (dae) => done(dae.scene),
      undefined,
      (err) => done(null, err),
    );
  } else if (ext === "stl") {
    new STLLoader(manager).load(
      path,
      (geom) => {
        const mat = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.6 });
        done(new THREE.Mesh(geom, mat));
      },
      undefined,
      (err) => done(null, err),
    );
  } else {
    done(null, new Error(`unsupported mesh extension: ${ext}`));
  }
};

// URDF has both DAE (visual) and STL (collision) refs. Force visual only.
urdfLoader.parseVisual = true;
urdfLoader.parseCollision = false;

function loadArm() {
  return new Promise((resolve, reject) => {
    urdfLoader.load("/nero_urdf", (robot) => resolve(robot), undefined, (err) => reject(err));
  });
}

async function buildArms() {
  const [leftRobot, rightRobot] = await Promise.all([loadArm(), loadArm()]);
  armRobots.left  = leftRobot;
  armRobots.right = rightRobot;

  // The Nero URDF's arm chain extends along its own +Z axis. World is now
  // Z-up, so an unrotated arm would also point straight up — wrong for a
  // side-mounted shoulder. We rotate each arm 90° around the world Y axis
  // so its chain becomes horizontal:
  //   right arm: Ry(+90°)  →  URDF +Z lands on world +X (robot's right).
  //   left  arm: Ry(-90°)  →  URDF +Z lands on world -X (robot's left).
  const rightMount = new THREE.Group();
  rightMount.position.set( SHOULDER_X, 0, SHOULDER_Z);
  // Ry(+90°) puts the chain along +X; additional Rx(180°) rolls the arm
  // 180° around its own (now world-X) chain axis so it sits right-side-up.
  rightMount.rotation.set(Math.PI, Math.PI / 2, 0);
  rightMount.add(rightRobot);
  scene.add(rightMount);

  const leftMount = new THREE.Group();
  leftMount.position.set(-SHOULDER_X, 0, SHOULDER_Z);
  leftMount.rotation.y = -Math.PI / 2;
  leftMount.add(leftRobot);
  scene.add(leftMount);

  applyArmPose("left");
  applyArmPose("right");
}

function applyArmPose(side) {
  const robot = armRobots[side];
  if (!robot) return;
  const cmd = sliderState[side].cmd;
  for (let i = 0; i < jointNames.length; i++) {
    const j = robot.joints[jointNames[i]];
    if (j) j.setJointValue(cmd[i]);
  }
}

// ------------ viewport sizing -------------------------------------------
function resizeViewport() {
  const w = Math.max(1, viewport.clientWidth);
  const h = Math.max(1, viewport.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resizeViewport);
new ResizeObserver(resizeViewport).observe(viewport);

function animate() {
  requestAnimationFrame(animate);
  resizeViewport();
  controls.update();
  renderer.render(scene, camera);
}

// ------------ sliders ---------------------------------------------------
function buildSliders() {
  for (const side of ["left", "right"]) {
    const container = panels[side].querySelector(".sliders");
    container.innerHTML = "";
    sliderState[side].cmd = new Array(jointNames.length).fill(0);
    sliderState[side].fb  = new Array(jointNames.length).fill(0);
    sliderState[side].rowEls = [];

    for (let i = 0; i < jointNames.length; i++) {
      const [lo, hi] = jointLimits[i];
      const row = document.createElement("div");
      row.className = "joint-row";
      row.innerHTML = `
        <label>${jointNames[i]}</label>
        <input type="range" min="${lo}" max="${hi}" step="0.005" value="0" />
        <span class="cmd">0.000</span>
        <span class="fb">fb –</span>
      `;
      const range   = row.querySelector("input");
      const cmdSpan = row.querySelector(".cmd");
      range.addEventListener("input", () => {
        const v = parseFloat(range.value);
        sliderState[side].cmd[i] = v;
        cmdSpan.textContent = v.toFixed(3);
        applyArmPose(side);
      });
      container.appendChild(row);
      sliderState[side].rowEls.push(row);
    }

    panels[side].querySelector(".btn-send").onclick = () => sendCommand(side);
    panels[side].querySelector(".btn-zero").onclick = () => zeroSliders(side);
    panels[side].querySelector(".btn-sync").onclick = () => syncFromRobot(side);
  }
}

function setSlider(side, i, value) {
  const row = sliderState[side].rowEls[i];
  if (!row) return;
  const range   = row.querySelector("input");
  const cmdSpan = row.querySelector(".cmd");
  const [lo, hi] = jointLimits[i];
  const v = Math.max(lo, Math.min(hi, value));
  range.value = v;
  sliderState[side].cmd[i] = v;
  cmdSpan.textContent = v.toFixed(3);
}

function zeroSliders(side) {
  for (let i = 0; i < jointNames.length; i++) setSlider(side, i, 0);
  applyArmPose(side);
}

function syncFromRobot(side) {
  const fb = sliderState[side].fb;
  for (let i = 0; i < jointNames.length; i++) setSlider(side, i, fb[i] ?? 0);
  applyArmPose(side);
}

function sendCommand(side) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ side, positions: sliderState[side].cmd }));
  }
}

function updateFeedbackDisplay(side, names, positions) {
  const byName = {};
  for (let i = 0; i < names.length; i++) byName[names[i]] = positions[i];
  for (let i = 0; i < jointNames.length; i++) {
    const v = byName[jointNames[i]];
    const row = sliderState[side].rowEls[i];
    if (!row) continue;
    const fbSpan = row.querySelector(".fb");
    if (typeof v === "number") {
      sliderState[side].fb[i] = v;
      fbSpan.textContent = "fb " + v.toFixed(3);
    } else {
      fbSpan.textContent = "fb –";
    }
  }
}

// ------------ websocket -------------------------------------------------
let ws = null;

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.addEventListener("open", () => {
    statusEl.textContent = "connected";
    statusEl.className = "status connected";
  });
  ws.addEventListener("close", () => {
    statusEl.textContent = "disconnected";
    statusEl.className = "status disconnected";
    setTimeout(connect, 1000);
  });
  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "state" && msg.data) {
      for (const side of ["left", "right"]) {
        const s = msg.data[side];
        if (s && s.names && s.positions) {
          updateFeedbackDisplay(side, s.names, s.positions);
        }
      }
    }
  });
}

// ------------ boot ------------------------------------------------------
async function boot() {
  const resp = await fetch("/joint_limits");
  const body = await resp.json();
  jointNames  = body.names;
  jointLimits = body.limits;
  buildSliders();
  resizeViewport();
  animate();
  connect();
  try {
    await buildArms();
  } catch (err) {
    console.error("URDF load failed", err);
    statusEl.textContent = "URDF load failed (check console)";
  }
}

boot().catch((err) => {
  console.error("boot failed", err);
  statusEl.textContent = "boot error";
});
