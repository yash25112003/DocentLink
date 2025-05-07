const mongoose = require('mongoose');
const crypto = require('crypto');

const userProfileSchema = new mongoose.Schema({
  // User Identification
  user_id: {
    type: String,
    required: true,
    unique: true,
    default: () => crypto.randomUUID()
  },

  // Basic User Information
  name: {
    type: String,
    required: true,
    trim: true,
    minlength: 2
  },
  email: {
    type: String,
    required: true,
    validate: {
      validator: v => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v),
      message: props => `${props.value} is not a valid email!`
    }
  },
  university: {
    type: String,
    required: true,
    enum: [
      "Arizona State University", 
      "Stanford University", 
      "MIT", 
      "Harvard University", 
      "UC Berkeley"
    ]
  },

  // Professor Selections
  professors: [{
    type: String,
    required: true
  }],

  // File Information
  file_name: {
    type: String,
    default: null
  },
  status: {
    type: String,
    enum: ['active', 'file_removed', 'pending'],
    default: 'pending'
  },

  // Resume Analysis Fields
  resume_analysis: {
    biography: {
      content: {
        type: String,
        description: "Biographical information from resume"
      },
      extractionStatus: {
        type: String,
        enum: ['success', 'failed', 'partial'],
        default: 'pending'
      },
      errorMessage: String
    },
    research_interests: [{
      type: String,
      description: "List of research interests"
    }],
    publications: [{
      title: {
        type: String,
        required: true
      },
      authors: [{
        type: String
      }],
      venue: String,
      year: Number,
      url: String,
      pdf: String,
      abstract: String,
      type: {
        type: String,
        enum: ['journal', 'conference', 'workshop', 'other'],
        default: 'other'
      },
      status: {
        type: String,
        enum: ['published', 'accepted', 'submitted', 'in_progress'],
        default: 'published'
      }
    }],
    awards: [{
      name: {
        type: String,
        required: true
      },
      year: Number,
      description: String,
      url: String,
      issuer: String,
      category: String
    }],
    education: [{
      degree: {
        type: String,
        required: true
      },
      institution: {
        type: String,
        required: true
      },
      year: Number,
      description: String,
      gpa: Number,
      major: String,
      minor: String,
      location: String,
      start_date: Date,
      end_date: Date,
      is_current: Boolean
    }],
    contact: {
      email: String,
      website: String,
      phone: String,
      location: String,
      linkedin: String,
      github: String,
      twitter: String,
      other_profiles: [{
        platform: String,
        url: String
      }]
    },
    skills: [{
      category: {
        type: String,
        enum: ['technical', 'soft', 'language', 'domain', 'other'],
        default: 'technical'
      },
      name: String,
      level: {
        type: String,
        enum: ['beginner', 'intermediate', 'advanced', 'expert'],
        default: 'intermediate'
      },
      years_experience: Number
    }],
    experience: [{
      title: String,
      company: String,
      start_date: Date,
      end_date: Date,
      description: String,
      technologies: [String],
      location: String,
      is_current: Boolean,
      type: {
        type: String,
        enum: ['full-time', 'part-time', 'internship', 'contract', 'freelance'],
        default: 'full-time'
      },
      highlights: [String],
      achievements: [String]
    }],
    projects: [{
      name: {
        type: String,
        required: true
      },
      description: String,
      technologies: [String],
      start_date: Date,
      end_date: Date,
      url: String,
      type: {
        type: String,
        enum: ['academic', 'personal', 'professional', 'research'],
        default: 'personal'
      },
      status: {
        type: String,
        enum: ['completed', 'in_progress', 'planned'],
        default: 'completed'
      },
      team_size: Number,
      role: String,
      highlights: [String],
      outcomes: [String]
    }],
    certifications: [{
      name: {
        type: String,
        required: true
      },
      issuer: String,
      date: Date,
      expiry_date: Date,
      credential_id: String,
      url: String,
      description: String
    }],
    languages: [{
      name: String,
      proficiency: {
        type: String,
        enum: ['basic', 'intermediate', 'advanced', 'native'],
        default: 'intermediate'
      }
    }],
    interests: [String],
    summary: {
      professional: String,
      academic: String,
      research: String
    },
    metadata: {
      last_updated: Date,
      extraction_version: String,
      confidence_score: Number,
      source_document: String
    }
  },

  // Timestamps
  created_at: {
    type: Date,
    default: Date.now,
    immutable: true
  },
  updated_at: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Pre-save middleware to update the updated_at timestamp
userProfileSchema.pre('save', function(next) {
  this.updated_at = new Date();
  next();
});

const UserProfile = mongoose.model('UserProfile', userProfileSchema);

module.exports = UserProfile; 