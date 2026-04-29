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
// Fetched from /torso_config at boot — single source of truth is the
// launch file, which passes the same values to both the xacro (URDF
// model) and the webapp node (this 3D rendering).
let TORSO_X = 0.185;   // width  (left↔right) — overwritten at boot
let TORSO_Y = 0.10;    // depth  (front↔back)
let TORSO_Z = 0.60;    // height (down↔up)
let SHOULDER_X = 0.112585; // lateral offset of shoulder from centerline
const SHOULDER_Z = 0.55; // shoulder height above floor
// Roll of each arm about its own +X axis (rad). Right arm rolls
// +SHOULDER_TILT, left rolls -SHOULDER_TILT. Set from /torso_config so
// the xacro/URDF and this rendering stay aligned.
let SHOULDER_TILT = 20.0 * Math.PI / 180.0;

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
const ghostRobots = { left: null, right: null };
// Tracks the last quest_status string per side so we can fire one-shot
// transitions (e.g. move the slider arm model to the calibration pose
// the instant CALIBRATING begins, not on every subsequent frame).
const prevQuestStatus = { left: "", right: "" };

// Quest controller marker. Drawn in ROBOT WORLD frame (scene root)
// at (anchor + dp_world), where:
//   anchor   = ghost's gripper_flange world position at calibration
//              (snapshotted once on every CALIBRATING entry)
//   dp_world = controller offset since calibration, published by
//              quest_leader_node on /<side>/quest_leader/controller_viz
// Orientation is the controller rotation since calibration (also
// world-frame). At calibration the marker sits at the ghost's flange
// with identity orientation; as the operator moves their hand it
// tracks the input the IK is consuming.
const controllerMarkers = { left: null, right: null }; // THREE.Group per side
const controllerAnchors = { left: null, right: null }; // THREE.Vector3 per side

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
  pika_gripper_description: "/pkg/pika_gripper_description",
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

function makeGhost(robot) {
  // Tint every mesh amber and make it translucent so the ghost reads as
  // a "target" overlay distinct from the real (feedback) arm.
  robot.traverse((obj) => {
    if (!obj.isMesh) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    obj.material = mats.map((m) => {
      const clone = m.clone();
      if ("color" in clone) clone.color = new THREE.Color(0xffb300);
      clone.transparent = true;
      clone.opacity = 0.35;
      clone.depthWrite = false;
      return clone;
    });
    if (!Array.isArray(obj.material)) obj.material = obj.material[0];
    obj.renderOrder = 500;
  });
  robot.visible = false;
  return robot;
}

// Per-side colour for the controller marker. Distinct from the amber
// ghost so all three (real arm, target ghost, controller input) read
// at a glance.
const CONTROLLER_COLOR = {
  left:  0x00ffaa, // greenish-cyan
  right: 0xff00aa, // magenta-pink
};

function buildControllerMarker(side) {
  // Called once calibration has driven the ghost to the calibration
  // pose. Snapshots the ghost's gripper_flange WORLD position as the
  // marker anchor, then builds a small group containing a coloured
  // sphere, an AxesHelper showing orientation, and a dashed tether
  // line back to the anchor (so the operator can see how far their
  // hand has drifted from the calibration point). All in the
  // robot-world frame (the scene root).
  const ghost = ghostRobots[side];
  if (!ghost) return;
  const flange = ghost.links && ghost.links["gripper_flange"];
  if (!flange) return;

  ghost.updateMatrixWorld(true);
  const eeWorld = new THREE.Vector3();
  flange.getWorldPosition(eeWorld);
  controllerAnchors[side] = eeWorld.clone();

  // Tear down any previous marker on this side.
  if (controllerMarkers[side]) {
    scene.remove(controllerMarkers[side]);
    controllerMarkers[side] = null;
  }

  const group = new THREE.Group();
  group.position.copy(eeWorld);

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.012, 16, 16),
    new THREE.MeshBasicMaterial({
      color: CONTROLLER_COLOR[side], depthWrite: false,
    }),
  );
  ball.renderOrder = 700;
  group.add(ball);

  // AxesHelper draws +X red, +Y green, +Z blue along the group's local
  // axes, so the marker's orientation is visible directly.
  const axes = new THREE.AxesHelper(0.10);
  axes.renderOrder = 700;
  group.add(axes);

  // Anchor pip at the calibration centre — fixed in world, so it sits
  // in the marker's local frame at -dp (updated each tick).
  const anchorPip = new THREE.Mesh(
    new THREE.SphereGeometry(0.005, 10, 10),
    new THREE.MeshBasicMaterial({
      color: CONTROLLER_COLOR[side], transparent: true,
      opacity: 0.5, depthWrite: false,
    }),
  );
  anchorPip.renderOrder = 700;
  group.add(anchorPip);

  // Tether line from marker centre to the anchor pip.
  const tetherGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, 0),
  ]);
  const tether = new THREE.Line(
    tetherGeom,
    new THREE.LineDashedMaterial({
      color: CONTROLLER_COLOR[side], dashSize: 0.01, gapSize: 0.006,
      transparent: true, opacity: 0.6, depthWrite: false,
    }),
  );
  tether.computeLineDistances();
  tether.renderOrder = 700;
  group.add(tether);

  group.userData.tether = tether;
  group.userData.anchorPip = anchorPip;

  scene.add(group);
  controllerMarkers[side] = group;
}

function updateControllerMarker(side, ctrl) {
  const marker = controllerMarkers[side];
  const anchor = controllerAnchors[side];
  if (!marker || !anchor || !ctrl) return;

  // Position = anchor + dp_world (publisher already maps Quest → world).
  marker.position.set(
    anchor.x + ctrl.px,
    anchor.y + ctrl.py,
    anchor.z + ctrl.pz,
  );
  // Orientation = dr_world. Set the marker's world quaternion to it
  // directly; AxesHelper will rotate with the group.
  marker.quaternion.set(ctrl.qx, ctrl.qy, ctrl.qz, ctrl.qw);

  // Tether endpoint in marker-local: inv(R_marker) · (anchor - pos).
  const tether = marker.userData.tether;
  const anchorPip = marker.userData.anchorPip;
  if (tether) {
    const anchorLocal = anchor.clone().sub(marker.position)
      .applyQuaternion(marker.quaternion.clone().invert());
    const positions = tether.geometry.attributes.position;
    positions.setXYZ(0, 0, 0, 0);
    positions.setXYZ(1, anchorLocal.x, anchorLocal.y, anchorLocal.z);
    positions.needsUpdate = true;
    tether.computeLineDistances();
    if (anchorPip) anchorPip.position.copy(anchorLocal);
  }
}

async function buildArms() {
  const [leftRobot, rightRobot, leftGhost, rightGhost] = await Promise.all([
    loadArm(), loadArm(), loadArm(), loadArm(),
  ]);
  armRobots.left  = leftRobot;
  armRobots.right = rightRobot;
  ghostRobots.left  = makeGhost(leftGhost);
  ghostRobots.right = makeGhost(rightGhost);

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
  // Additional roll about the arm's own +X axis (intrinsic), matching
  // the xacro's shoulder_tilt_deg. rotateX composes onto the current
  // rotation so the tilt is applied in the arm's local frame.
  rightMount.rotateX(SHOULDER_TILT);
  rightMount.add(rightRobot);
  rightMount.add(rightGhost);
  scene.add(rightMount);

  const leftMount = new THREE.Group();
  leftMount.position.set(-SHOULDER_X, 0, SHOULDER_Z);
  leftMount.rotation.y = -Math.PI / 2;
  leftMount.rotateX(-SHOULDER_TILT);
  leftMount.add(leftRobot);
  leftMount.add(leftGhost);
  scene.add(leftMount);

  applyArmPose("left");
  applyArmPose("right");

  // Show base_link coordinate axes on each arm so the user can
  // identify which axis points "down" for gravity compensation.
  // RED = +X, GREEN = +Y, BLUE = +Z.
  for (const side of ["left", "right"]) {
    const robot = armRobots[side];
    if (!robot) continue;
    const axes = new THREE.AxesHelper(0.20);
    axes.renderOrder = 998;
    robot.add(axes);
    // Label each axis tip.
    const labels = [
      { text: "+X", color: "#ff3333", pos: [0.22, 0, 0] },
      { text: "+Y", color: "#33ff33", pos: [0, 0.22, 0] },
      { text: "+Z", color: "#3366ff", pos: [0, 0, 0.22] },
    ];
    for (const { text, color, pos } of labels) {
      const sprite = makeTextSprite(text, color);
      sprite.position.set(...pos);
      sprite.scale.set(0.12, 0.06, 1);
      sprite.renderOrder = 999;
      robot.add(sprite);
    }
  }
}

function applyGhostPose(side, names, positions) {
  const robot = ghostRobots[side];
  if (!robot || !names || !positions) return;
  for (let i = 0; i < names.length; i++) {
    const j = robot.joints[names[i]];
    if (j) j.setJointValue(positions[i]);
  }
}

function applyArmPose(side) {
  const robot = armRobots[side];
  if (!robot) return;
  const cmd = sliderState[side].cmd;
  for (let i = 0; i < jointNames.length; i++) {
    const j = robot.joints[jointNames[i]];
    if (j) j.setJointValue(cmd[i]);
  }
  // Drive the two prismatic gripper fingers from the "gripper" width value.
  // Pika gripper: both jaws move along Y axis.
  //   gripper_joint1 (left):  limit [-0.05, 0], negative = close
  //   gripper_joint2 (right): limit [0, 0.05],  positive = open
  const gi = jointNames.indexOf("gripper");
  if (gi >= 0) {
    const w = cmd[gi];  // width in metres (0–0.1)
    const j1 = robot.joints["gripper_joint1"];
    const j2 = robot.joints["gripper_joint2"];
    if (j1) j1.setJointValue(-w * 0.5);  // left jaw (negative Y = close)
    if (j2) j2.setJointValue(w * 0.5);   // right jaw (positive Y = open)
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

    const gcBtn = panels[side].querySelector(".btn-grav-comp");
    if (gcBtn) {
      gcBtn.onclick = async () => {
        const enabling = !gcBtn.classList.contains("active");
        const resp = await fetch("/gravity_comp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ side, enabled: enabling }),
        });
        if (resp.ok) {
          gcBtn.classList.toggle("active", enabling);
          gcBtn.textContent = enabling ? "Gravity Comp ON" : "Gravity Comp";
        } else {
          const err = await resp.json().catch(() => ({}));
          console.error("gravity_comp toggle failed:", err);
        }
      };
    }

    // --- Quest leader buttons --------------------------------------------
    const qCalib   = panels[side].querySelector(".btn-quest-calibrate");
    const qPreview = panels[side].querySelector(".btn-quest-preview");
    const qFollow  = panels[side].querySelector(".btn-quest-follow");

    async function postQuest(path, body) {
      const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return r.json();
    }

    if (qCalib) {
      qCalib.onclick = async () => {
        qCalib.disabled = true;
        qCalib.textContent = "Calibrating...";
        try {
          const d = await postQuest("/quest_leader/calibrate", { side });
          if (!d.ok) alert("Calibrate failed: " + (d.message || d.error || "?"));
        } finally {
          qCalib.disabled = false;
          qCalib.textContent = "Calibrate Quest";
        }
      };
    }
    if (qPreview) {
      qPreview.onclick = async () => {
        const enabling = !qPreview.classList.contains("active");
        const d = await postQuest("/quest_leader/preview", { side, enabled: enabling });
        if (d.ok) {
          qPreview.classList.toggle("active", enabling);
          qPreview.textContent = enabling ? "Preview ON" : "Preview";
          if (!enabling && qFollow) {
            qFollow.classList.remove("active");
            qFollow.textContent = "Follow";
          }
        } else {
          alert("Preview failed: " + (d.message || d.error || "?"));
        }
      };
    }
    if (qFollow) {
      qFollow.onclick = async () => {
        const enabling = !qFollow.classList.contains("active");
        const d = await postQuest("/quest_leader/follow", { side, enabled: enabling });
        if (d.ok) {
          qFollow.classList.toggle("active", enabling);
          qFollow.textContent = enabling ? "Follow ON" : "Follow";
        } else {
          alert("Follow failed: " + (d.message || d.error || "?"));
        }
      };
    }

    const teachBtn = panels[side].querySelector(".btn-teach");
    if (teachBtn) {
      teachBtn.onclick = async () => {
        const enabling = !teachBtn.classList.contains("active");
        teachBtn.textContent = enabling ? "Enabling..." : "Disabling...";
        try {
          const resp = await fetch("/teach_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ side, enabled: enabling }),
          });
          const data = await resp.json();
          if (data.ok) {
            teachBtn.classList.toggle("active", enabling);
            teachBtn.textContent = enabling ? "Teach ON" : "Teach";
          } else {
            teachBtn.textContent = teachBtn.classList.contains("active") ? "Teach ON" : "Teach";
            console.error("teach_mode failed:", data.error);
            alert("Teach mode failed: " + (data.error || "unknown error"));
          }
        } catch (err) {
          teachBtn.textContent = teachBtn.classList.contains("active") ? "Teach ON" : "Teach";
          console.error("teach_mode error:", err);
          alert("Teach mode error: " + err.message);
        }
      };
    }
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
        if (s && s.quest_ghost && s.quest_ghost.names && s.quest_ghost.positions) {
          applyGhostPose(side, s.quest_ghost.names, s.quest_ghost.positions);
        }
        if (s && s.quest_controller) {
          updateControllerMarker(side, s.quest_controller);
        }
        if (s && s.quest_status !== undefined) {
          const statusEl = document.querySelector(`.quest-status[data-side="${side}"]`);
          const state = (s.quest_status || "").toString();
          const ghost = ghostRobots[side];
          if (ghost) ghost.visible = state.includes("preview=True");
          // On the transition into CALIBRATING, push the calibration joint
          // pose (carried on the ghost topic and broadcast immediately by
          // the node) into the slider arm model. Follow-on: if the Follow
          // button is active, also send it to the real arm.
          const prev = prevQuestStatus[side];
          const enteringCalibrate =
            state.includes("CALIBRATING") && !prev.includes("CALIBRATING");
          if (enteringCalibrate && s.quest_ghost &&
              s.quest_ghost.names && s.quest_ghost.positions) {
            const byName = {};
            for (let i = 0; i < s.quest_ghost.names.length; i++) {
              byName[s.quest_ghost.names[i]] = s.quest_ghost.positions[i];
            }
            for (let i = 0; i < jointNames.length; i++) {
              const v = byName[jointNames[i]];
              if (typeof v === "number") setSlider(side, i, v);
            }
            applyArmPose(side);
            const followBtn = document.querySelector(
              `.btn-quest-follow[data-side="${side}"]`);
            if (followBtn && followBtn.classList.contains("active")) {
              sendCommand(side);
            }
            // Ghost joints were just updated — anchor the controller
            // marker at the new calibration EE position.
            buildControllerMarker(side);
          }
          prevQuestStatus[side] = state;
          if (statusEl) {
            // Prominent countdown during CALIBRATING.
            const m = state.match(/countdown=(\d+)/);
            if (state.includes("CALIBRATING") && m) {
              statusEl.innerHTML =
                `<span style="font-size:20px;font-weight:bold;color:#ffb300">` +
                `Hold Quest controller steady — ${m[1]}</span>`;
            } else if (state.includes("CALIBRATING")) {
              statusEl.textContent = "Calibrating…";
            } else {
              statusEl.textContent = "quest: " + state;
            }
          }
          // Enable/disable Preview + Send based on state.
          const ready = state.includes("READY") || state.includes("ACTIVE");
          const pv = document.querySelector(`.btn-quest-preview[data-side="${side}"]`);
          const fl = document.querySelector(`.btn-quest-follow[data-side="${side}"]`);
          if (pv) pv.disabled = !ready;
          if (fl) fl.disabled = !(pv && pv.classList.contains("active"));
          // Auto-reflect server-side auto-disarm of Follow.
          if (fl && fl.classList.contains("active") && !state.includes("follow=True")) {
            fl.classList.remove("active");
            fl.textContent = "Follow";
          }
        }
      }
    }
  });
}

// ------------ WebRTC camera streams -------------------------------------
// Server publishes up to three video tracks per peer connection in this
// fixed order: fisheye, color, depth. We create one recvonly transceiver
// per expected track, send an SDP offer to /offer, and pair each incoming
// track with its <video> element by the server-returned "cameras" list.

// Fixed display order for camera tiles. Matches WebappNode.camera_slots()
// on the server (left arm first, right arm second; fisheye, color, depth
// within each side). The server can trim any subset out of its /offer
// answer when that camera isn't connected — the client uses the
// server's returned `cameras` list to pair incoming tracks with the
// correct video element regardless of which ones are disabled.
const CAM_ORDER = [
  "fisheye_left", "color_left", "depth_left",
  "fisheye_right", "color_right", "depth_right",
];

function setCamStatus(name, text, cls) {
  const el = document.getElementById("cam-status-" + name);
  if (!el) return;
  el.textContent = text;
  el.className = "cam-status " + (cls || "");
}

// Latest RTCPeerConnection (kept module-scope so the stats poller can
// call getStats() without re-negotiating).
let cameraPc = null;
// Track which remote track belongs to which camera name, keyed by trackId.
const trackIdToCamera = new Map();

async function startCameras() {
  for (const n of CAM_ORDER) setCamStatus(n, "negotiating", "pending");

  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });
  cameraPc = pc;

  // One recvonly video transceiver per possible camera — server trims
  // unused ones in the answer.
  const transceivers = CAM_ORDER.map(() =>
    pc.addTransceiver("video", { direction: "recvonly" })
  );

  // Collect incoming tracks in arrival order. We pair them with the
  // server's camera list (returned in the answer) below.
  const incomingTracks = [];
  pc.addEventListener("track", (ev) => {
    incomingTracks.push(ev.track);
  });

  pc.addEventListener("connectionstatechange", () => {
    if (pc.connectionState === "failed" || pc.connectionState === "closed") {
      for (const n of CAM_ORDER) setCamStatus(n, pc.connectionState, "failed");
    }
  });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const resp = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    console.error("WebRTC /offer failed", resp.status, text);
    for (const n of CAM_ORDER) setCamStatus(n, "offer failed", "failed");
    return;
  }
  const answer = await resp.json();
  const serverCameras = answer.cameras || [];
  await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });

  // Cameras the server didn't add aren't going to fire a track event.
  for (const n of CAM_ORDER) {
    if (!serverCameras.includes(n)) setCamStatus(n, "disabled", "failed");
  }

  // Wait briefly for all expected tracks to arrive, then pair them in
  // arrival order with the server's camera name list.
  const deadline = Date.now() + 5000;
  while (incomingTracks.length < serverCameras.length && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  for (let i = 0; i < serverCameras.length; i++) {
    const name = serverCameras[i];
    const track = incomingTracks[i];
    const videoEl = document.getElementById("video-" + name);
    if (!track || !videoEl) {
      setCamStatus(name, "no track", "failed");
      continue;
    }
    const stream = new MediaStream([track]);
    videoEl.srcObject = stream;
    trackIdToCamera.set(track.id, name);
    videoEl.addEventListener(
      "playing",
      () => setCamStatus(name, "live", "live"),
      { once: true },
    );
  }

  // Start the stats poller once the PC is built.
  if (!window.__camStatsPollerStarted) {
    window.__camStatsPollerStarted = true;
    pollCameraStats();
  }
}

// ---- wall clock (for visual latency compare against frame overlay) -----
function formatClock(ms) {
  const d = new Date(ms);
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  return (
    pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds()) +
    "." + pad(d.getMilliseconds(), 3)
  );
}
function tickClock() {
  const el = document.getElementById("wallclock");
  if (el) el.textContent = formatClock(Date.now());
  requestAnimationFrame(tickClock);
}
tickClock();

// ---- per-camera stats polling -----------------------------------------
// Every second, fetch /stats for capture-side numbers and pc.getStats()
// for decode/receive-side numbers, then update each camera's stats line.
const prevInbound = new Map();  // trackName -> {bytes, frames, t}

async function pollCameraStats() {
  while (true) {
    await new Promise((r) => setTimeout(r, 1000));
    let serverStats = null;
    try {
      const resp = await fetch("/stats", { cache: "no-store" });
      serverStats = await resp.json();
    } catch (_) {}

    const peerStats = cameraPc ? await cameraPc.getStats() : null;
    const perCam = new Map();
    if (peerStats) {
      peerStats.forEach((rep) => {
        if (rep.type !== "inbound-rtp" || rep.kind !== "video") return;
        const name = trackIdToCamera.get(rep.trackIdentifier) ||
                     trackIdToCamera.get(rep.trackId);
        if (!name) return;
        perCam.set(name, rep);
      });
    }

    for (const name of CAM_ORDER) {
      const el = document.getElementById("cam-stats-" + name);
      if (!el) continue;
      const srv = serverStats && serverStats.cameras &&
                  serverStats.cameras[name];
      const rep = perCam.get(name);
      let line = "";
      if (srv && srv.shape) {
        const [h, w] = srv.shape;
        line += `cap ${srv.fps.toFixed(1)} fps ${w}x${h}`;
      } else {
        line += "cap –";
      }
      if (rep) {
        const now = performance.now();
        const prev = prevInbound.get(name);
        prevInbound.set(name, {
          bytes: rep.bytesReceived || 0,
          frames: rep.framesDecoded || 0,
          t: now,
        });
        let fps = 0, kbps = 0;
        if (prev) {
          const dt = (now - prev.t) / 1000;
          if (dt > 0) {
            fps = ((rep.framesDecoded || 0) - prev.frames) / dt;
            kbps = ((rep.bytesReceived || 0) - prev.bytes) * 8 / 1000 / dt;
          }
        }
        const dropped = rep.framesDropped || 0;
        const lost = rep.packetsLost || 0;
        const jitterMs = Math.round((rep.jitter || 0) * 1000);
        line += ` · rx ${fps.toFixed(1)} fps ${kbps.toFixed(0)} kbps` +
                ` · drop ${dropped} · lost ${lost} · jit ${jitterMs}ms`;
        el.classList.toggle("warn", lost > 5 || dropped > 10);
      } else {
        line += " · rx –";
      }
      el.textContent = line;
    }
  }
}

// ------------ boot ------------------------------------------------------
async function boot() {
  // Fetch torso dimensions from the server (single source of truth).
  try {
    const tcResp = await fetch("/torso_config");
    const tc = await tcResp.json();
    TORSO_X = tc.torso_width;
    TORSO_Y = tc.torso_depth;
    TORSO_Z = tc.torso_height;
    SHOULDER_X = TORSO_X / 2.0;
    if (typeof tc.shoulder_tilt_deg === "number") {
      SHOULDER_TILT = tc.shoulder_tilt_deg * Math.PI / 180.0;
    }
    // Update the torso box + shoulder mounts that were created with
    // the initial (possibly stale) defaults.
    torso.geometry.dispose();
    torso.geometry = new THREE.BoxGeometry(TORSO_X, TORSO_Y, TORSO_Z);
    torso.position.set(0, 0, TORSO_Z / 2);
    controls.target.set(0, 0, TORSO_Z - 0.05);
    controls.update();
  } catch (e) {
    console.warn("torso_config fetch failed, using defaults:", e);
  }

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
  // WebRTC negotiation is independent of the URDF/WS path; don't let
  // camera failures break the arm UI.
  startCameras().catch((err) => {
    console.error("WebRTC start failed", err);
    for (const n of CAM_ORDER) setCamStatus(n, "error", "failed");
  });
}

boot().catch((err) => {
  console.error("boot failed", err);
  statusEl.textContent = "boot error";
});
