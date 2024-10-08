// static/script.js
/**
 * Handles the multi-step form navigation, form submission, and result display via modal.
 */

const form = document.getElementById('loanForm');
const resultModal = document.getElementById('resultModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const modalDetails = document.getElementById('modalDetails');
const steps = document.querySelectorAll('.form-step');
const progressSteps = document.querySelectorAll('.progress-step');

/**
 * Navigates to the next step in the form.
 * @param {number} currentStep - The current step number.
 */
function nextStep(currentStep) {
  if (currentStep < steps.length) {
    steps[currentStep - 1].classList.remove('active');
    steps[currentStep].classList.add('active');
    progressSteps[currentStep].classList.add('active');
  }
}

/**
 * Navigates to the previous step in the form.
 * @param {number} currentStep - The current step number.
 */
function prevStep(currentStep) {
  if (currentStep > 1) {
    steps[currentStep - 1].classList.remove('active');
    steps[currentStep - 2].classList.add('active');
    progressSteps[currentStep - 1].classList.remove('active');
  }
}

/**
 * Toggles the visibility of the goods price input based on purchase decision.
 */
function toggleGoodsPrice() {
  const buyGoods = document.getElementById('buy_goods').value;
  const goodsPriceGroup = document.getElementById('goodsPriceGroup');
  goodsPriceGroup.style.display = buyGoods === 'true' ? 'block' : 'none';
  document.getElementById('amt_goods_price').required = buyGoods === 'true';
  if (buyGoods !== 'true') {
    document.getElementById('amt_goods_price').value = 0;
  }
}

/**
 * Submits the loan application data to the backend for prediction.
 */
function submitApplication() {
  if (validateForm()) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert buy_goods to boolean
    data.buy_goods = data.buy_goods === 'true';

    // Convert Age from years to negative days
    const ageYears = parseInt(data.days_birth, 10);
    data.days_birth = -Math.abs(ageYears * 365);

    // Convert monetary fields to floats
    const monetaryFields = [
      'amt_income_total',
      'amt_credit',
      'amt_goods_price',
    ];
    monetaryFields.forEach((field) => {
      data[field] = parseFloat(data[field]);
    });

    // Convert loan_term_years to integer
    data.loan_term_years = parseInt(data.loan_term_years, 10);

    // Show loading modal
    showLoadingModal();

    $.ajax({
      url: '/predict',
      type: 'POST',
      contentType: 'application/json',
      data: JSON.stringify(data),
      success: function (response) {
        displayResult(response);
      },
      error: function (xhr, status, error) {
        console.error('Error:', error);
        console.error('Status:', status);
        console.error('Response:', xhr.responseText);
        let errorMsg = 'An error occurred.';
        if (xhr.responseText) {
          try {
            const errorResponse = JSON.parse(xhr.responseText);
            if (errorResponse.detail) {
              errorMsg = Array.isArray(errorResponse.detail)
                ? errorResponse.detail.map((err) => err.msg).join(' ')
                : errorResponse.detail;
            }
          } catch (e) {
            console.error('Error parsing error response:', e);
          }
        }
        showErrorModal(errorMsg);
      },
    });
  }
}

/**
 * Validates the form inputs before submission.
 * @returns {boolean} - Returns true if the form is valid, false otherwise.
 */
function validateForm() {
  const requiredFields = form.querySelectorAll('[required]');
  for (const field of requiredFields) {
    if (!field.value) {
      alert(`Please fill out the ${field.name.replace('_', ' ')} field.`);
      field.focus();
      return false;
    }
  }

  const ageYears = parseInt(document.getElementById('days_birth').value, 10);
  if (isNaN(ageYears) || ageYears < 18 || ageYears > 100) {
    alert('Please enter a valid age between 18 and 100.');
    document.getElementById('days_birth').focus();
    return false;
  }

  const monetaryFields = ['amt_income_total', 'amt_credit', 'amt_goods_price'];
  for (const field of monetaryFields) {
    const value = parseFloat(document.getElementById(field).value);
    if (isNaN(value) || value < 0) {
      alert(`Please enter a valid amount for ${field.replace('_', ' ')}.`);
      document.getElementById(field).focus();
      return false;
    }
  }

  const loanTermYears = parseInt(
    document.getElementById('loan_term_years').value,
    10
  );
  if (isNaN(loanTermYears) || loanTermYears < 1 || loanTermYears > 30) {
    alert('Please enter a valid loan term between 1 and 30 years.');
    document.getElementById('loan_term_years').focus();
    return false;
  }

  return true;
}

/**
 * Displays the loan application result in a modal.
 * @param {Object} prediction - The prediction response from the backend.
 */
function displayResult(prediction) {
  // Hide loading modal
  hideModal();

  const percentage = (prediction.probability_of_default * 100).toFixed(2);
  let title = '';
  let message = '';
  let details = '';

  if (prediction.loan_approval === 'Approved') {
    title = '<i class="fas fa-check-circle"></i> Loan Approved';
    message = `Congratulations! Your loan application has been approved.`;
    details = `
            <p><strong>Monthly Payment:</strong> $${prediction.monthly_payment.toLocaleString()}</p>
            <p><strong>Probability of Default:</strong> ${percentage}%</p>
            <p><strong>Risk Level:</strong> ${prediction.risk_level}</p>
        `;
  } else if (prediction.loan_approval === 'Rejected') {
    title = '<i class="fas fa-times-circle"></i> Loan Rejected';
    message = `We're sorry, but your loan application was rejected.`;
    details = `
            <p><strong>Monthly Payment:</strong> $${prediction.monthly_payment.toLocaleString()}</p>
            <p><strong>Probability of Default:</strong> ${percentage}%</p>
            <p><strong>Reason:</strong> ${
              prediction.rejection_reason || 'Risk level classified as High.'
            }</p>
        `;
  }

  if (
    prediction.anomalies_detected &&
    prediction.anomalies_details.length > 0
  ) {
    details += `<h4>Anomalies Detected:</h4><ul>`;
    prediction.anomalies_details.forEach((anomaly) => {
      details += `<li><strong>${
        anomaly.feature
      }:</strong> $${anomaly.value.toLocaleString()} (Allowed: $${anomaly.lower_bound.toLocaleString()} - $${anomaly.upper_bound.toLocaleString()})</li>`;
    });
    details += `</ul>`;
  }

  modalTitle.innerHTML = title;
  modalMessage.innerHTML = message;
  modalDetails.innerHTML = details;

  // Style modal based on approval status
  if (prediction.loan_approval === 'Approved') {
    resultModal.querySelector(
      '.modal-content'
    ).style.borderTop = `5px solid var(--success-color)`;
  } else {
    resultModal.querySelector(
      '.modal-content'
    ).style.borderTop = `5px solid var(--error-color)`;
  }

  // Show modal
  resultModal.style.display = 'flex';
}

/**
 * Displays an error modal with a specific message.
 * @param {string} message - The error message to display.
 */
function showErrorModal(message) {
  console.log('Showing error modal with message:', message);
  hideModal(); // Hide any existing modal
  modalTitle.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
  modalMessage.innerHTML = message;
  modalDetails.innerHTML = '';
  resultModal.querySelector('.modal-content').style.borderTop =
    '5px solid var(--error-color)';
  resultModal.style.display = 'flex';
  console.log('Error modal should now be visible');
  console.log('Modal display style:', resultModal.style.display);
  console.log(
    'Modal visibility:',
    window.getComputedStyle(resultModal).visibility
  );
  console.log('Modal opacity:', window.getComputedStyle(resultModal).opacity);
}

/**
 * Shows a loading modal while the request is being processed.
 */
function showLoadingModal() {
  modalTitle.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing';
  modalMessage.innerHTML = 'Please wait while we process your application.';
  modalDetails.innerHTML = '';

  resultModal.querySelector(
    '.modal-content'
  ).style.borderTop = `5px solid var(--primary-color)`;

  // Show modal
  resultModal.style.display = 'flex';
}

/**
 * Closes the result modal.
 */
function closeModal() {
  resultModal.style.display = 'none';
  // Optionally, reset the form upon successful submission
  // Uncomment the following line if desired
  // form.reset();
}

/**
 * Closes the modal when clicking outside the modal content.
 */
window.onclick = function (event) {
  if (event.target == resultModal) {
    resultModal.style.display = 'none';
  }
};
