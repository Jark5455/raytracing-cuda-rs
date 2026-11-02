pub struct Interval {
    pub min: f32,
    pub max: f32,
}

impl Interval {
    pub fn size(&self) -> f32 {
        self.max - self.min
    }

    pub fn contains(&self, value: f32) -> bool {
        self.min <= value && value <= self.max
    }

    pub fn surrounds(&self, value: f32) -> bool {
        self.min < value && value < self.max
    }

    pub fn clamp(&self, value: f32) -> f32 {
        if value < self.min {
            self.min
        } else if value > self.max {
            self.max
        } else {
            value
        }
    }
}

const EMPTY: Interval = Interval { min: f32::INFINITY, max: f32::NEG_INFINITY };
const UNIVERSE: Interval = Interval { min: f32::NEG_INFINITY, max: f32::INFINITY };