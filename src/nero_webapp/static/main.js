// Simple dual-Nero control UI.
//
// - Builds a stick-figure Three.js scene: torso box + two 7-joint arms.
// - Opens a WebSocket to /ws, receives 20 Hz feedback state, sends
//   commanded positions on Send-button clicks.
// - Sliders are built from /joint_limits.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ------------ approximate Nero kinematics (stick-figure only) ---------
// This is NOT the real URDF. It's a visual proxy so that slider
// movements give immediate feedback; the ROS side has the real URDF.
// Each entry: rotation axis applied at the joint, then a bone segment
// offset to the NEXT joint (in that joint's frame after rotation).
const ARM_CHAIN = [
  { axis: "z", bone: [0.00, 0.00, 0.15] }, // joint1 shoulder yaw
  { axis: "y", bone: [0.00, 0.00, 0.10] }, // joint2 shoulder pitch
  { axis: "z", bone: [0.00, 0.00, 0.30] }, // joint3 upper-arm roll
  { axis: "y", bone: [0.00, 0.00, 0.05] }, // joint4 elbow
  { axis: "z", bone: [0.00, 0.00, 0.30] }, // joint5 forearm roll
  { axis: "y", bone: [0.00, 0.00, 0.05] }, // joint6 wrist pitch
  { axis: "z", bone: [0.00, 0.00, 0.12] }, // joint7 wrist roll (tool)
];

// Torso box matches two_nero.urdf.xacro defaults (0.10 x 0.30 x 0.60).
const TORSO_SIZE = [0.10, 0.30, 0.60];
// Shoulder mount offsets on torso in meters.
const SHOULDER_Y = 0.15;
const SHOULDER_Z = 0.22;

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

// ------------ three.js scene --------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090c);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
camera.position.set(1.4, 1.1, 1.4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.3, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(2, 3, 2);
scene.add(sun);

// Floor grid for orientation.
const grid = new THREE.GridHelper(4, 16, 0x2a2f37, 0x1a1d22);
scene.add(grid);

// Torso.
const torsoMat = new THREE.MeshStandardMaterial({ color: 0x3b4252, roughness: 0.6 });
const torso = new THREE.Mesh(
  new THREE.BoxGeometry(TORSO_SIZE[0], TORSO_SIZE[2], TORSO_SIZE[1]),
  torsoMat,
);
// Torso centered at z=TORSO_SIZE[2]/2 above floor.
torso.position.set(0, TORSO_SIZE[2] / 2, 0);
scene.add(torso);

// Build one stick-figure arm. Returns an array of joint Groups so we
// can set .rotation on them from the slider values.
function buildArm(colorHex) {
  const jointMat = new THREE.MeshStandardMaterial({ color: 0x5aa9ff, roughness: 0.4 });
  const boneMat  = new THREE.MeshStandardMaterial({ color: colorHex,  roughness: 0.5 });

  const root = new THREE.Group();
  let parent = root;
  const jointGroups = [];

  for (const { bone } of ARM_CHAIN) {
    const jg = new THREE.Group();
    parent.add(jg);
    jointGroups.push(jg);

    // Joint sphere rendered at jg origin.
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 12), jointMat);
    jg.add(sphere);

    // Bone cylinder from (0,0,0) to bone[] in the joint's local frame.
    const len = Math.hypot(bone[0], bone[1], bone[2]);
    if (len > 1e-4) {
      const cyl = new THREE.Mesh(
        new THREE.CylinderGeometry(0.014, 0.014, len, 10),
        boneMat,
      );
      // CylinderGeometry is along +Y by default. Rotate so it points along bone vector.
      const dir = new THREE.Vector3(bone[0], bone[1], bone[2]).normalize();
      const up = new THREE.Vector3(0, 1, 0);
      const q = new THREE.Quaternion().setFromUnitVectors(up, dir);
      cyl.quaternion.copy(q);
      cyl.position.set(bone[0] / 2, bone[1] / 2, bone[2] / 2);
      jg.add(cyl);
    }

    // Child frame starts at bone tip.
    const tip = new THREE.Group();
    tip.position.set(bone[0], bone[1], bone[2]);
    jg.add(tip);
    parent = tip;
  }

  return { root, jointGroups };
}

// Instantiate left/right arms and mount them on the torso's sides.
const armMeshes = {
  left:  buildArm(0x9ece6a),
  right: buildArm(0xf7768e),
};

// Left shoulder: +Y side of torso (from world), arm root aimed outward (+Y).
armMeshes.left.root.position.set(0, SHOULDER_Z + TORSO_SIZE[2] / 2 - TORSO_SIZE[2] / 2, SHOULDER_Y);
// Actually: place at torso top-left. Torso center is at y=TORSO_SIZE[2]/2 in WORLD y (up).
// We want shoulder near torso top. Simpler: mount at world (0, SHOULDER_Z_world, ±SHOULDER_Y).
armMeshes.left.root.position.set(0, SHOULDER_Z, SHOULDER_Y);
// Rotate arm root so its local +Z (default down-the-chain) points outward +Z world = +Z (sideways).
// Our chain is built along +Z local. To point arm sideways (+Z world → +Y world for left),
// rotate -90° around X: local +Z becomes world +Y.
armMeshes.left.root.rotation.x = -Math.PI / 2;
scene.add(armMeshes.left.root);

armMeshes.right.root.position.set(0, SHOULDER_Z, -SHOULDER_Y);
// Rotate +90° around X so local +Z → world -Y.
armMeshes.right.root.rotation.x = Math.PI / 2;
scene.add(armMeshes.right.root);

// Apply current slider-cmd values to the three.js joints.
function applyArmPose(side) {
  const groups = armMeshes[side].jointGroups;
  const pos = sliderState[side].cmd;
  for (let i = 0; i < groups.length && i < pos.length; i++) {
    const axis = ARM_CHAIN[i].axis;
    const g = groups[i];
    g.rotation.set(0, 0, 0);
    if (axis === "x") g.rotation.x = pos[i];
    if (axis === "y") g.rotation.y = pos[i];
    if (axis === "z") g.rotation.z = pos[i];
  }
}

function resizeViewport() {
  const w = Math.max(1, viewport.clientWidth);
  const h = Math.max(1, viewport.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resizeViewport);
// Observe the container too — grid/flex layout changes don't always fire window resize.
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
      const range = row.querySelector("input");
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
  applyArmPose("left");
  applyArmPose("right");
}

function setSlider(side, i, value) {
  const row = sliderState[side].rowEls[i];
  if (!row) return;
  const range = row.querySelector("input");
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
}

boot().catch((err) => {
  console.error("boot failed", err);
  statusEl.textContent = "boot error";
});
