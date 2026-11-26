import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const nativeOnly = document.body.dataset.view === "native";
const sceneHost = $("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
sceneHost.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 500);
camera.position.set(2.5, 1.6, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.autoRotateSpeed = 0.85;
controls.enablePan = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8d847b, 1.7));
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(3, 4, 6);
scene.add(light);
if (!nativeOnly) scene.add(new THREE.AxesHelper(0.16));
if (nativeOnly) $("point-size").value = "3";

let trace = null;
let frames = [];
let frameIndex = 0;
let selectedToken = null;
let selectedStripe = null;
let shownOrder = null;
let rankColors = [];
let pointCloud = null;
let representativeCloud = null;
let selectedLines = null;
let meshObject = null;
let meshDepthObject = null;
let meshObjectURL = null;
let autoplay = null;
let radius = 0.012;
let worldSpan = 2;
const sphere = new THREE.SphereGeometry(1, 8, 6);
const pointMat = new THREE.MeshBasicMaterial();
const repMat = new THREE.MeshBasicMaterial({ color: 0xf3a038 });
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dummy = new THREE.Object3D();
const defaultCam = new THREE.Vector3(2.5, 1.6, 3);
const defaultTarget = new THREE.Vector3();
const gradient = [0x243caa, 0x405ed9, 0x248bc1, 0x10a987, 0x80bf43, 0xeaa527]
  .map(hex => new THREE.Color(hex));
const dimmed = new THREE.Color("#abaeaf");
const plainPoint = new THREE.Color(nativeOnly ? "#315b92" : "#506e9b");
const surfaceAllowance = { value: 0 };
const meshDepthMaterial = new THREE.MeshBasicMaterial({ colorWrite: false });
meshDepthMaterial.onBeforeCompile = shader => {
  // Move only the depth pass along the viewing ray to keep a thin shell
  // of predictions visible on and just behind the triangulated surface.
  shader.uniforms.surfaceAllowance = surfaceAllowance;
  shader.vertexShader = "uniform float surfaceAllowance;\n" + shader.vertexShader.replace(
    "#include <project_vertex>",
    `#include <project_vertex>
     mvPosition.xyz += normalize(mvPosition.xyz) * surfaceAllowance;
     gl_Position = projectionMatrix * mvPosition;`
  );
};

function status(message, isError = false) {
  $("status").textContent = message;
  $("status").style.color = isError ? "#b34132" : "#785d47";
}

function colorForGroup(id) {
  if (!Number.isInteger(id) || id < 0) return new THREE.Color("#9c9489");
  const hue = ((id * 0.61803398875) % 1 + 1) % 1;
  return new THREE.Color().setHSL(hue, 0.55, 0.48);
}

function prepareRankColors(count) {
  rankColors = Array.from({ length: count }, (_, rank) => {
    const offset = rank * (gradient.length - 1) / Math.max(1, count - 1);
    const left = Math.min(Math.floor(offset), gradient.length - 2);
    return gradient[left].clone().lerp(gradient[left + 1], offset - left);
  });
}

function validateOrder(order, count) {
  if (!Array.isArray(order) || order.length !== count || new Set(order).size !== count ||
      order.some(i => !Number.isInteger(i) || i < 0 || i >= count)) {
    throw new Error("排序数组须为原始 token 索引的完整排列");
  }
}

function validate(input) {
  if (!input || input.schema_version !== 1 || !Array.isArray(input.frames) || !input.frames.length) {
    throw new Error("需要 schema_version=1 的 merge-trajectory.json");
  }
  const tokenCount = input.token_count;
  if (!Number.isInteger(tokenCount) || tokenCount < 1) throw new Error("token_count 无效");
  const seen = new Set();
  for (const f of input.frames) {
    if (!Number.isInteger(f.step) || seen.has(f.step)) throw new Error("Step 缺失或重复");
    seen.add(f.step);
    if (!Array.isArray(f.predicted_positions) || f.predicted_positions.length !== tokenCount) {
      throw new Error("Step " + f.step + " 的位置形状错误");
    }
    if (!f.predicted_positions.every(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite))) {
      throw new Error("Step " + f.step + " 的位置包含 NaN 或无效值");
    }
    if (nativeOnly) continue;
    if (f.prediction_morton_order != null) validateOrder(f.prediction_morton_order, tokenCount);
    if (!f.layers || typeof f.layers !== "object") throw new Error("Missing layer snapshots");
    for (const [layer, snap] of Object.entries(f.layers)) {
      if (!/^\d+$/.test(layer) || !Array.isArray(snap.group_ids) ||
          snap.group_ids.length !== tokenCount || !Number.isInteger(snap.group_count)) {
        throw new Error("Step " + f.step + " Layer " + layer + " 的分组数据不完整");
      }
      if (snap.group_ids.some(g => !Number.isInteger(g) || g < 0 || g >= snap.group_count)) {
        throw new Error("存在越界的 group id");
      }
      if (snap.representatives && (snap.representatives.length !== snap.group_count ||
          snap.representatives.some(r => !Number.isInteger(r) || r < 0 || r >= tokenCount))) {
        throw new Error("代表 token 索引无效");
      }
      if (snap.token_order != null) {
        validateOrder(snap.token_order, tokenCount);
        if (!Number.isInteger(snap.stripe_count) || snap.stripe_count < 1 ||
            tokenCount % snap.stripe_count !== 0) throw new Error("排序分段的数量无效");
      }
    }
  }
  if (nativeOnly && (input.source?.method !== "native" || seen.size !== 50 ||
      input.frames.some(f => f.step < 0 || f.step >= 50))) {
    throw new Error("需要完整的 50 步 Native 位置预测");
  }
}

function mortonData(frame, snap, displayedFrame) {
  const actual = displayedFrame.step !== frame.step && snap?.operator === "position_stripe" &&
    Boolean(snap.token_order);
  const tokens = actual ? snap.token_order : displayedFrame.prediction_morton_order;
  if (!tokens) return null;
  const ranks = new Uint32Array(tokens.length);
  tokens.forEach((token, rank) => { ranks[token] = rank; });
  return { ranks, stripes: actual ? snap.stripe_count : 0, actual };
}

function positionData(frame, snap) {
  const source = $("position-source").value;
  if (source === "used" && snap && snap.position_origin_step != null) {
    const previous = frames.find(f => f.step === snap.position_origin_step);
    if (previous) {
      return {
        frame: previous,
        note: "当前层使用第 " + (previous.step + 1) + " 步的预测坐标。"
      };
    }
    return { frame,
      note: "没有找到用于合并的原始位置记录，当前显示同一步的位置预测（仅供参考）。" };
  }
  if (snap && snap.position_origin_step == null) {
    return { frame,
      note: "该方法不使用预测位置进行匹配；这里只把当前 step 的坐标作为视觉参考。"};
  }
  return { frame,
    note: "当前 step 的 clean-state 预测位置；它不一定是当前层实际用于分组的位置。" };
}

function removeObject(object) {
  if (!object) return;
  scene.remove(object);
  if (object.geometry && object !== pointCloud && object !== representativeCloud) object.geometry.dispose();
}

function resetCamera(positions) {
  const box = new THREE.Box3();
  for (const p of positions) box.expandByPoint(new THREE.Vector3(...p));
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  worldSpan = Math.max(size.x, size.y, size.z, 0.2);
  defaultTarget.copy(center);
  const viewOffset = new THREE.Vector3(1.2, 0.65, 1.48);
  defaultCam.copy(center).addScaledVector(viewOffset, worldSpan * (nativeOnly ? 0.78 : 1));
  controls.target.copy(defaultTarget);
  camera.position.copy(defaultCam);
  camera.near = Math.max(0.0005, worldSpan / 2500);
  camera.far = worldSpan * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function setPoints(positions, snap, order) {
  const n = positions.length;
  if (!pointCloud || pointCloud.count !== n) {
    if (pointCloud) scene.remove(pointCloud);
    pointCloud = new THREE.InstancedMesh(sphere, pointMat, n);
    pointCloud.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    pointCloud.frustumCulled = false;
    scene.add(pointCloud);
  }
  radius = worldSpan * 0.0045 * Number($("point-size").value) / 4;
  const useRank = $("color-mode").value === "morton" && order;
  const stripeWidth = order?.stripes ? n / order.stripes : 0;
  const activeStripe = stripeWidth > 0 && selectedStripe != null;
  const selectedGroup = snap && selectedToken != null ? snap.group_ids[selectedToken] : null;
  for (let i = 0; i < n; ++i) {
    const group = snap ? snap.group_ids[i] : i;
    const highlighted = activeStripe ? Math.floor(order.ranks[i] / stripeWidth) === selectedStripe :
      selectedGroup != null && group === selectedGroup;
    const hasSelection = activeStripe || selectedGroup != null;
    const hidden = $("only-group").checked && hasSelection && !highlighted;
    dummy.position.set(...positions[i]);
    dummy.scale.setScalar(hidden ? 0 : radius * (highlighted ? 1.55 : 1));
    dummy.updateMatrix();
    pointCloud.setMatrixAt(i, dummy.matrix);
    let color = useRank ? rankColors[order.ranks[i]] : snap ? colorForGroup(group) : plainPoint;
    if (hasSelection && !highlighted) color = dimmed;
    if (i === selectedToken) {
      color = useRank ? color : new THREE.Color("#e34932");
      dummy.scale.setScalar(radius * 2.1);
      dummy.updateMatrix();
      pointCloud.setMatrixAt(i, dummy.matrix);
    }
    pointCloud.setColorAt(i, color);
  }
  pointCloud.instanceMatrix.needsUpdate = true;
  // InstancedMesh.raycast first checks this sphere; matrices change every frame.
  pointCloud.computeBoundingSphere();
  if (pointCloud.instanceColor) pointCloud.instanceColor.needsUpdate = true;
  pointCloud.visible = $("show-tokens").checked;
  if (nativeOnly) return;

  if (representativeCloud) scene.remove(representativeCloud);
  const representatives = snap?.representatives || [];
  representativeCloud = new THREE.InstancedMesh(sphere, repMat, Math.max(1, representatives.length));
  representativeCloud.count = representatives.length;
  representativeCloud.frustumCulled = false;
  for (let j = 0; j < representatives.length; ++j) {
    const index = representatives[j];
    const hide = (!$('show-reps').checked || $('only-group').checked) &&
      selectedGroup != null && j !== selectedGroup;
    dummy.position.set(...positions[index]);
    dummy.scale.setScalar(hide ? 0 : radius * 2.2);
    dummy.updateMatrix();
    representativeCloud.setMatrixAt(j, dummy.matrix);
  }
  representativeCloud.instanceMatrix.needsUpdate = true;
  representativeCloud.visible = $("show-reps").checked || selectedGroup != null;
  scene.add(representativeCloud);

  removeObject(selectedLines);
  selectedLines = null;
  if (snap && selectedGroup != null && $("show-lines").checked) {
    const members = [];
    for (let i = 0; i < n; ++i) if (snap.group_ids[i] === selectedGroup) members.push(i);
    const center = new THREE.Vector3();
    if (snap.representatives) {
      center.set(...positions[snap.representatives[selectedGroup]]);
    } else {
      for (const idx of members) center.add(new THREE.Vector3(...positions[idx]));
      center.multiplyScalar(1 / members.length);
    }
    const lineCoords = [];
    for (const idx of members) lineCoords.push(...positions[idx], center.x, center.y, center.z);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineCoords, 3));
    selectedLines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
      color: 0xd55c37, transparent: true, opacity: 0.82, depthTest: false
    }));
    selectedLines.renderOrder = 10;
    scene.add(selectedLines);
  }
}

function drawOrder() {
  const canvas = $("order-strip");
  const order = shownOrder;
  const show = Boolean(order && $("color-mode").value === "morton");
  $("order-panel").hidden = !show;
  if (!show) return;
  const n = rankColors.length;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  for (let x = 0; x < width; x++) {
    ctx.fillStyle = "#" + rankColors[Math.min(n - 1, Math.floor(x * n / width))].getHexString();
    ctx.fillRect(x, 0, 1, height);
  }
  const selectedRank = selectedToken == null ? null : order.ranks[selectedToken];
  const stripe = selectedStripe ?? (selectedRank != null && order.stripes ?
    Math.floor(selectedRank * order.stripes / n) : null);
  if (order.stripes) {
    ctx.fillStyle = "rgba(9, 19, 43, .38)";
    for (let i = 1; i < order.stripes; i++) {
      const notch = i % 16 === 0 ? height * .44 : height * .23;
      ctx.fillRect(Math.round(i * width / order.stripes), height - notch, Math.round(dpr), notch);
    }
    if (stripe != null) {
      const span = width / order.stripes;
      ctx.strokeStyle = "white";
      ctx.lineWidth = dpr;
      ctx.strokeRect(stripe * span, dpr, span, height - 2 * dpr);
    }
  }
  if (selectedRank != null) {
    ctx.fillStyle = "white";
    ctx.fillRect((selectedRank + .5) * width / n - dpr, 0, 2 * dpr, height);
  }
  $("order-source").textContent = order.actual ?
    "实际分段 · " + order.stripes + " 段 × " + n / order.stripes + " token" : "预测坐标排序";
  canvas.classList.toggle("interactive", Boolean(order.stripes));
  $("order-selection").textContent = stripe != null ?
    "第 " + (stripe + 1) + " 段 · 排名 " + (stripe * n / order.stripes + 1) + "–" +
      ((stripe + 1) * n / order.stripes) + (selectedRank != null ? " · 选中排名 " + (selectedRank + 1) : "") :
    selectedRank != null ? "第 " + (selectedRank + 1) + " 名 · token #" + selectedToken :
    order.actual ? "点击色带查看同一分段。每段内部还会进行特征匹配。" :
    "颜色表示这一帧预测坐标的排序名次。";
}

function updateMetrics(frame, snap, note) {
  $("stat-step").textContent = frame.step + 1;
  $("stat-n").textContent = trace.token_count;
  if (nativeOnly) {
    $("selection-info").textContent = selectedToken == null ? "点击三维点查看预测坐标。" :
      "Token #" + selectedToken + " · XYZ (" +
        frame.predicted_positions[selectedToken].map(value => value.toFixed(4)).join(", ") + ")";
    return;
  }
  $("stat-layer").textContent = $("layer-select").disabled ? "—" : $("layer-select").value;
  $("stat-k").textContent = snap ? snap.group_count : trace.token_count;
  $("position-note").textContent = note;
  if (snap) {
    const reduced = ((1 - snap.group_count / trace.token_count) * 100).toFixed(1);
    const planFrom = snap.plan_origin_step === frame.step ? "新建" : "缓存自第 " + (snap.plan_origin_step + 1) + " 步";
    $("plan-info").textContent = snap.operator + " · " + snap.assignment_kind +
      " · 减少 " + reduced + "% token · " + planFrom +
      (snap.assignment_kind === "soft" ? " · 主导权重着色（非硬合并）；argmax 占用组 " +
        (snap.occupied_primary_groups ?? new Set(snap.group_ids).size) + "/" + snap.group_count : "");
  } else {
    $("plan-info").textContent = "本 step/layer 未记录合并计划：可能为完整计算、未启用合并，或诊断时过滤了该 layer。";
  }
  if (selectedStripe != null) {
    $("selection-info").textContent = "第 " + (selectedStripe + 1) + " 个 Morton 段已高亮。";
    return;
  }
  if (!snap || selectedToken == null) {
    $("selection-info").textContent = selectedToken == null ? "点击三维点查看 token。" : "Token #" + selectedToken;
    return;
  }
  const group = snap.group_ids[selectedToken];
  let members = 0;
  for (const g of snap.group_ids) if (g === group) members++;
  const rep = snap.representatives ? "实际代表 token #" + snap.representatives[group] :
              "HardPlan 无明确代表 token（绘制组中心）";
  let detail = "token #" + selectedToken + " → group #" + group +
    "；组内 token：" + members + " 个；" + rep + "。";
  if (snap.confidence) {
    detail += " 本 token 的最大权重 " + Number(snap.confidence[selectedToken]).toFixed(4);
    if (snap.second_group_ids && snap.second_group_ids[selectedToken] >= 0)
      detail += "；次大权重来自组 #" + snap.second_group_ids[selectedToken] +
                "（" + Number(snap.second_weights[selectedToken]).toFixed(4) + "）。";
  }
  $("selection-info").textContent = detail;
}

function drawFrame() {
  if (!trace) return;
  const frame = frames[frameIndex];
  const snap = nativeOnly ? null : frame.layers?.[$("layer-select").value] || null;
  const { frame: displayedFrame, note } = positionData(frame, snap);
  const positions = displayedFrame.predicted_positions;
  shownOrder = nativeOnly ? null : mortonData(frame, snap, displayedFrame);
  if (selectedToken != null && selectedToken >= positions.length) selectedToken = null;
  setPoints(positions, snap, shownOrder);
  drawOrder();
  updateMetrics(frame, snap, note);
  $("step-slider").value = frameIndex;
  $("step-select").value = String(frameIndex);
}

async function loadTrace(input, name) {
  validate(input);
  stopPlayback();
  trace = input;
  if (!nativeOnly) prepareRankColors(trace.token_count);
  frames = [...input.frames].sort((a, b) => a.step - b.step);
  frameIndex = frames.length - 1;
  selectedToken = null;
  selectedStripe = null;
  $("case-title").textContent = name || input.source?.run_id || "Trajectory";
  $("method-label").textContent = nativeOnly ? "Native" : input.source?.method || "unknown method";
  $("step-slider").max = frames.length - 1;
  $("step-select").replaceChildren();
  for (const [index, f] of frames.entries()) {
    const opt = document.createElement("option");
    opt.value = String(index);
    opt.textContent = "第 " + (f.step + 1) + " 步";
    $("step-select").append(opt);
  }
  const layerIds = new Set();
  for (const f of frames) for (const layer of Object.keys(f.layers || {})) layerIds.add(Number(layer));
  for (const layer of input.source?.layers ?? []) layerIds.add(layer);
  $("layer-select").replaceChildren();
  for (const layer of [...layerIds].sort((a, b) => a - b)) {
    const opt = document.createElement("option");
    opt.value = String(layer);
    opt.textContent = "Layer " + layer;
    $("layer-select").append(opt);
  }
  const firstRecordedLayer = [...layerIds].sort((a, b) => a - b)
    .find(l => frames.some(f => f.layers?.[String(l)]));
  $("layer-select").value = firstRecordedLayer?.toString() ?? ($("layer-select").options[0]?.value || "");
  $("layer-select").disabled = firstRecordedLayer == null;
  resetCamera(frames[frameIndex].predicted_positions);
  drawFrame();
  status("已加载 " + frames.length + " 个去噪步。");
}

function stopPlayback() {
  if (autoplay) window.clearInterval(autoplay);
  autoplay = null;
  $("play").textContent = "▶ 播放";
}

function changeFrame(index) {
  frameIndex = Math.max(0, Math.min(frames.length - 1, index));
  selectedToken = null;
  selectedStripe = null;
  drawFrame();
}

function syncMeshVisibility() {
  const showMesh = Boolean(meshObject) && $("show-mesh").checked;
  $("show-mesh").disabled = !meshObject;
  $("show-all-points").disabled = !showMesh;
  if (meshObject) meshObject.visible = showMesh;
  if (meshDepthObject) meshDepthObject.visible = showMesh && !$("show-all-points").checked;
}

function replaceMesh(nextMesh) {
  if (meshDepthObject) scene.remove(meshDepthObject);
  meshDepthObject = null;
  if (meshObject) {
    scene.remove(meshObject);
    const geometries = new Set();
    const materials = new Set();
    meshObject.traverse(obj => {
      if (!obj.isMesh) return;
      geometries.add(obj.geometry);
      materials.add(obj.material);
    });
    geometries.forEach(geometry => geometry.dispose());
    materials.forEach(material => material.dispose());
  }
  meshObject = nextMesh;
  if (!meshObject) {
    syncMeshVisibility();
    return;
  }
  const meshMaterial = new THREE.MeshStandardMaterial({
    color: nativeOnly ? 0xb8c9de : 0xccc4b9,
    roughness: 0.86, metalness: 0, transparent: true,
    opacity: nativeOnly ? 0.25 : 0.35,
    side: nativeOnly ? THREE.FrontSide : THREE.DoubleSide,
    depthWrite: false,
  });
  meshObject.traverse(obj => {
    if (obj.isMesh) obj.material = meshMaterial;
  });
  if (nativeOnly) {
    const span = new THREE.Box3().setFromObject(meshObject).getSize(new THREE.Vector3());
    surfaceAllowance.value = 0.015 * Math.max(span.x, span.y, span.z);
    meshDepthObject = meshObject.clone();
    meshDepthObject.traverse(obj => {
      if (!obj.isMesh) return;
      obj.material = meshDepthMaterial;
      obj.renderOrder = -1;
    });
    scene.add(meshDepthObject);
  }
  scene.add(meshObject);
  syncMeshVisibility();
}

$("trajectory-file").addEventListener("change", async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  try { await loadTrace(JSON.parse(await file.text()), file.name); }
  catch (err) { status("轨迹读取失败：" + err.message, true); }
});
$("mesh-file").addEventListener("change", async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  if (meshObjectURL) URL.revokeObjectURL(meshObjectURL);
  meshObjectURL = URL.createObjectURL(file);
  try {
    replaceMesh((await new GLTFLoader().loadAsync(meshObjectURL)).scene);
    status("已加载参考网格。");
  }
  catch (err) { status("Mesh 读取失败：" + err.message, true); }
});
$("step-slider").addEventListener("input", e => { stopPlayback(); changeFrame(Number(e.target.value)); });
$("step-select").addEventListener("change", e => { stopPlayback(); changeFrame(Number(e.target.value)); });
for (const control of ["layer-select", "position-source", "color-mode"]) {
  $(control).addEventListener("change", () => { selectedStripe = null; drawFrame(); });
}
$("order-strip").addEventListener("click", event => {
  if (!shownOrder?.stripes) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const stripe = Math.min(shownOrder.stripes - 1, Math.floor((event.clientX - rect.left) / rect.width * shownOrder.stripes));
  stopPlayback();
  selectedToken = null;
  selectedStripe = selectedStripe === stripe ? null : stripe;
  drawFrame();
});
for (const key of ["show-tokens", "show-reps", "show-lines", "only-group", "point-size"]) {
  $(key).addEventListener("input", drawFrame);
}
for (const key of ["show-mesh", "show-all-points"]) $(key).addEventListener("change", syncMeshVisibility);
$("auto-rotate").addEventListener("change", e => { controls.autoRotate = e.target.checked; });
$("play").addEventListener("click", () => {
  if (!trace) return;
  if (autoplay) { stopPlayback(); return; }
  if (frameIndex === frames.length - 1) changeFrame(0);
  $("play").textContent = "❚❚ 暂停";
  autoplay = window.setInterval(() => {
    if (frameIndex >= frames.length - 1) stopPlayback();
    else changeFrame(frameIndex + 1);
  }, 380);
});
$("reset").addEventListener("click", () => {
  if (!trace) return;
  const reference = nativeOnly ? frames.at(-1) : frames[frameIndex];
  resetCamera(reference.predicted_positions);
  drawFrame();
});
$("screenshot").addEventListener("click", () => {
  renderer.render(scene, camera);
  const view = renderer.domElement;
  const showOrder = !$("order-panel").hidden;
  const margin = 24;
  const output = document.createElement("canvas");
  output.width = view.width + 2 * margin;
  output.height = view.height + (showOrder ? 130 : 80);
  const ctx = output.getContext("2d");
  ctx.fillStyle = "#f7f8fb";
  ctx.fillRect(0, 0, output.width, output.height);
  ctx.fillStyle = "#202c43";
  ctx.font = "600 17px sans-serif";
  const step = trace ? frames[frameIndex].step + 1 : null;
  const layer = $("layer-select").disabled ? "" : " · Layer " + $("layer-select").value;
  ctx.fillText($("case-title").textContent + (step != null ? " · 第 " + step + " 步" + layer : ""), margin, 34);
  ctx.drawImage(view, margin, 54);
  if (showOrder) {
    const top = view.height + 62;
    ctx.drawImage($("order-strip"), margin, top, view.width, 16);
    ctx.font = "14px sans-serif";
    ctx.fillText("Morton 首位", margin, top + 37);
    ctx.fillText($("order-source").textContent, margin, top + 57);
    ctx.textAlign = "right";
    ctx.fillText("末位", output.width - margin, top + 37);
  }
  const link = document.createElement("a");
  link.download = nativeOnly ? "native-position-step-" + (step ?? "empty") + ".png" :
    "merge-step-" + (step ?? "empty") + "-layer-" + $("layer-select").value + ".png";
  link.href = output.toDataURL("image/png");
  link.click();
});
renderer.domElement.addEventListener("pointerdown", e => {
  renderer.domElement.dataset.downX = String(e.clientX);
  renderer.domElement.dataset.downY = String(e.clientY);
});
renderer.domElement.addEventListener("pointerup", event => {
  if (!trace || !pointCloud || !$("show-tokens").checked) return;
  const dx = event.clientX - Number(renderer.domElement.dataset.downX);
  const dy = event.clientY - Number(renderer.domElement.dataset.downY);
  if (dx * dx + dy * dy > 36) return; // Orbit drag, not a click.
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
              -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  let hit = raycaster.intersectObject(pointCloud, false)[0];
  if (hit && meshDepthObject?.visible) {
    const surface = raycaster.intersectObject(meshObject, true)[0];
    if (surface && hit.distance > surface.distance + surfaceAllowance.value) hit = null;
  }
  stopPlayback();
  selectedStripe = null;
  selectedToken = hit && hit.instanceId != null ? hit.instanceId : null;
  drawFrame();
});

function resize() {
  const w = Math.max(1, sceneHost.clientWidth), h = Math.max(1, sceneHost.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(sceneHost);
new ResizeObserver(drawOrder).observe($("order-panel"));
resize();
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

async function loadBundledTrace(entry) {
  const trajectory = fetch(entry.trajectory, { cache: "no-cache" }).then(response => {
    if (!response.ok) throw new Error("轨迹文件 HTTP " + response.status);
    return response.json();
  });
  const reference = entry.mesh ? new GLTFLoader().loadAsync(entry.mesh).catch(error => error) : null;
  const [payload, mesh] = await Promise.all([trajectory, reference]);
  await loadTrace(payload, entry.name || "Research case");
  replaceMesh(mesh?.scene ?? null);
  if (mesh instanceof Error) status("位置轨迹已加载，参考网格读取失败。", true);
}

(async () => {
  try {
    const response = await fetch("./data/manifest.json", { cache: "no-cache" });
    if (!response.ok) throw new Error("No bundled manifest");
    const manifest = await response.json();
    await loadBundledTrace(manifest);
    if (nativeOnly && manifest.cases?.length > 1) {
      const picker = $("case-select");
      manifest.cases.forEach((entry, index) => picker.add(new Option(entry.name, String(index))));
      $("case-picker").hidden = false;
      let loadedCase = picker.value;
      picker.addEventListener("change", async () => {
        stopPlayback();
        picker.disabled = true;
        status("正在加载位置轨迹…");
        try {
          await loadBundledTrace(manifest.cases[Number(picker.value)]);
          loadedCase = picker.value;
        } catch (err) {
          picker.value = loadedCase;
          status("样例读取失败：" + err.message, true);
        } finally {
          picker.disabled = false;
        }
      });
    }
  } catch (err) {
    status(nativeOnly ? "位置轨迹读取失败：" + err.message :
      "未加载内置轨迹。可导入 merge-trajectory.json。" +
      (err.message === "No bundled manifest" ? "" : " " + err.message), nativeOnly);
  }
})();
