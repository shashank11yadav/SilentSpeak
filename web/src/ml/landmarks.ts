// Mirrors pipeline/src/silentspeak/landmarks.py. Indices refer to MediaPipe's
// 478-point face mesh; the parity fixture checks both lists match.

export const LIP_INDICES = [
  0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
  269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
] as const;

export const EYE_OUTER = [33, 263] as const;
export const NOSE_TIP = 1;
export const CHIN = 152;

export const STORED_INDICES: readonly number[] = [...LIP_INDICES, ...EYE_OUTER, NOSE_TIP, CHIN];
export const N_STORED = STORED_INDICES.length; // 44
export const N_LIPS = LIP_INDICES.length; // 40

export const LIP_MIDLINE = [0, 13, 14, 17] as const;

export const LIP_PAIRS: readonly (readonly [number, number])[] = [
  [61, 291],
  [78, 308],
  [185, 409],
  [40, 270],
  [39, 269],
  [37, 267],
  [146, 375],
  [91, 321],
  [181, 405],
  [84, 314],
  [191, 415],
  [80, 310],
  [81, 311],
  [82, 312],
  [95, 324],
  [88, 318],
  [178, 402],
  [87, 317],
];

export const INNER_UPPER = 13;
export const INNER_LOWER = 14;

// Contours for drawing, as face-mesh indices from one mouth corner to the other.
export const OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291] as const;
export const OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291] as const;
export const INNER_UPPER_CONTOUR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308] as const;
export const INNER_LOWER_CONTOUR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308] as const;

export const BLENDSHAPE_NAMES = [
  '_neutral',
  'browDownLeft',
  'browDownRight',
  'browInnerUp',
  'browOuterUpLeft',
  'browOuterUpRight',
  'cheekPuff',
  'cheekSquintLeft',
  'cheekSquintRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'eyeLookDownLeft',
  'eyeLookDownRight',
  'eyeLookInLeft',
  'eyeLookInRight',
  'eyeLookOutLeft',
  'eyeLookOutRight',
  'eyeLookUpLeft',
  'eyeLookUpRight',
  'eyeSquintLeft',
  'eyeSquintRight',
  'eyeWideLeft',
  'eyeWideRight',
  'jawForward',
  'jawLeft',
  'jawOpen',
  'jawRight',
  'mouthClose',
  'mouthDimpleLeft',
  'mouthDimpleRight',
  'mouthFrownLeft',
  'mouthFrownRight',
  'mouthFunnel',
  'mouthLeft',
  'mouthLowerDownLeft',
  'mouthLowerDownRight',
  'mouthPressLeft',
  'mouthPressRight',
  'mouthPucker',
  'mouthRight',
  'mouthRollLower',
  'mouthRollUpper',
  'mouthShrugLower',
  'mouthShrugUpper',
  'mouthSmileLeft',
  'mouthSmileRight',
  'mouthStretchLeft',
  'mouthStretchRight',
  'mouthUpperUpLeft',
  'mouthUpperUpRight',
  'noseSneerLeft',
  'noseSneerRight',
] as const;
export const N_BLENDSHAPES = BLENDSHAPE_NAMES.length; // 52

export const BLEND_SINGLE = [
  'jawOpen',
  'mouthClose',
  'mouthFunnel',
  'mouthPucker',
  'mouthRollLower',
  'mouthRollUpper',
  'mouthShrugLower',
  'mouthShrugUpper',
] as const;

export const BLEND_PAIRS = [
  ['mouthLeft', 'mouthRight'],
  ['mouthSmileLeft', 'mouthSmileRight'],
  ['mouthFrownLeft', 'mouthFrownRight'],
  ['mouthDimpleLeft', 'mouthDimpleRight'],
  ['mouthStretchLeft', 'mouthStretchRight'],
  ['mouthPressLeft', 'mouthPressRight'],
  ['mouthLowerDownLeft', 'mouthLowerDownRight'],
  ['mouthUpperUpLeft', 'mouthUpperUpRight'],
] as const;

const ROW = new Map(STORED_INDICES.map((idx, row) => [idx, row]));

/** Row of a face-mesh index in the stored [N_STORED, 3] array. */
export function storedRow(meshIndex: number): number {
  const row = ROW.get(meshIndex);
  if (row === undefined) throw new Error(`face-mesh index ${meshIndex} is not stored`);
  return row;
}
