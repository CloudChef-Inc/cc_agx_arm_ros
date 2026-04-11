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

// ------------ torso + mount geometry (matches two_nero.urdf.xacro) ------
const TORSO_SIZE  = [0.10, 0.30, 0.60]; // depth (x), width (y), height (z)
const SHOULDER_Y  = 0.15; // lateral offset from centerline
const SHOULDER_Z  = 0.55; // height above floor at the shoulder mount

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
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090c);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
camera.position.set(1.8, 1.5, 1.8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(2, 3, 2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.3);
fill.position.set(-2, 1, -2);
scene.add(fill);

// Floor grid.
scene.add(new THREE.GridHelper(4, 16, 0x2a2f37, 0x1a1d22));

// Torso box to give the arms a home.
const torso = new THREE.Mesh(
  new THREE.BoxGeometry(TORSO_SIZE[0], TORSO_SIZE[2], TORSO_SIZE[1]),
  new THREE.MeshStandardMaterial({ color: 0x2a2f37, roughness: 0.7 }),
);
torso.position.set(0, TORSO_SIZE[2] / 2, 0);
scene.add(torso);

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

  // URDF base is Z-up; three.js scene is Y-up. Wrap each robot in a Group
  // that does the Z-up → Y-up axis swap and positions it as a shoulder.
  // Rotating -90° around X maps URDF +Z (up) to three.js +Y (up).
  const leftMount = new THREE.Group();
  leftMount.rotation.x = -Math.PI / 2;
  leftMount.position.set(0, SHOULDER_Z, SHOULDER_Y);
  // Point the arm outward to the left (three.js +Y after the axis swap).
  leftMount.rotateZ(Math.PI / 2);
  leftMount.add(leftRobot);
  scene.add(leftMount);

  const rightMount = new THREE.Group();
  rightMount.rotation.x = -Math.PI / 2;
  rightMount.position.set(0, SHOULDER_Z, -SHOULDER_Y);
  rightMount.rotateZ(-Math.PI / 2);
  rightMount.add(rightRobot);
  scene.add(rightMount);

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
